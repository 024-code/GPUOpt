from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable
from uuid import UUID

from uuid import uuid4 as _new_uuid

from pydantic import BaseModel as _PydanticBase

from .schemas import (
    AlertRecord,
    AlertRule,
    AnalysisSummary,
    ApprovalRecord,
    ApprovalStep,
    BaselineInfo,
    ClusterCreate,
    ClusterRecord,
    ClusterStateData,
    EnvironmentCheckReport,
    ActuationRecord,
    NotificationChannel,
    NotificationMessage,
    PolicyRule,
    RecommendationSet,
    RecommendationStatus,
    ResourceRecommendation,
    TwinState,
    WorkloadAnalysisResult,
)


class RepositoryError(RuntimeError):
    pass


# ── Database Backends ──────────────────────────────────────────


class DatabaseBackend(ABC):
    """Pluggable database backend (SQLite / PostgreSQL)."""

    @abstractmethod
    @contextmanager
    def connect(self) -> Generator[Any, None, None]:
        ...

    @abstractmethod
    def initialize(self) -> None:
        ...


class SqliteBackend(DatabaseBackend):
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._lock = threading.RLock()

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.database_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._lock, self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            try:
                conn.execute("ALTER TABLE clusters ADD COLUMN region TEXT")
            except Exception:
                pass


class PostgresBackend(DatabaseBackend):
    def __init__(self, database_url: str, min_conn: int = 2, max_conn: int = 10) -> None:
        import psycopg2
        from psycopg2 import pool as pgpool
        from psycopg2.extras import RealDictCursor

        self._pool = pgpool.ThreadedConnectionPool(min_conn, max_conn, database_url)
        self._cursor_factory = RealDictCursor
        self._lock = threading.RLock()

    @contextmanager
    def connect(self) -> Generator[Any, None, None]:
        from psycopg2 import extensions as pg_extensions

        conn = self._pool.getconn()
        conn.autocommit = False
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            try:
                self._pool.putconn(conn)
            except Exception:
                pass

    def initialize(self) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(POSTGRES_INIT_SQL)

    def close(self) -> None:
        self._pool.closeall()


def create_backend(
    database_url: str,
    database_path: Path | None = None,
    pool_min: int = 2,
    pool_max: int = 10,
) -> DatabaseBackend:
    if database_url.startswith("postgres"):
        return PostgresBackend(database_url, pool_min, pool_max)
    if database_url.startswith("sqlite"):
        if database_path is not None:
            return SqliteBackend(database_path)
        path_str = database_url.replace("sqlite:///", "")
        return SqliteBackend(Path(path_str))
    raise RepositoryError(f"Unsupported database URL scheme: {database_url}")


# ── Schema SQL ──────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS clusters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    environment TEXT NOT NULL,
    connector_type TEXT NOT NULL,
    description TEXT,
    kube_context TEXT,
    kubeconfig_path TEXT,
    in_cluster INTEGER NOT NULL,
    credential_ref TEXT,
    options_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS check_reports (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    overall_status TEXT NOT NULL,
    report_json TEXT NOT NULL,
    FOREIGN KEY(cluster_id) REFERENCES clusters(id)
);

CREATE INDEX IF NOT EXISTS idx_reports_cluster_completed
ON check_reports(cluster_id, completed_at DESC);

CREATE TABLE IF NOT EXISTS cluster_state (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    state_json TEXT NOT NULL,
    FOREIGN KEY(cluster_id) REFERENCES clusters(id)
);

CREATE INDEX IF NOT EXISTS idx_state_cluster_collected
ON cluster_state(cluster_id, collected_at DESC);

CREATE TABLE IF NOT EXISTS baselines (
    cluster_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    set_at TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    node_count INTEGER NOT NULL DEFAULT 0,
    gpu_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(cluster_id) REFERENCES clusters(id)
);

CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    FOREIGN KEY(cluster_id) REFERENCES clusters(id)
);

CREATE INDEX IF NOT EXISTS idx_analyses_cluster_gen
ON analyses(cluster_id, generated_at DESC);

CREATE TABLE IF NOT EXISTS recommendations (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    recs_json TEXT NOT NULL,
    FOREIGN KEY(cluster_id) REFERENCES clusters(id)
);

CREATE INDEX IF NOT EXISTS idx_recs_cluster_gen
ON recommendations(cluster_id, generated_at DESC);

CREATE TABLE IF NOT EXISTS twins (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL UNIQUE,
    synced_at TEXT NOT NULL,
    state_json TEXT NOT NULL,
    FOREIGN KEY(cluster_id) REFERENCES clusters(id)
);

CREATE TABLE IF NOT EXISTS actuations (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    rec_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    dry_run INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    actions_json TEXT NOT NULL DEFAULT '[]',
    result_summary TEXT DEFAULT '',
    error_message TEXT DEFAULT '',
    rollback_of TEXT DEFAULT '',
    rolled_back_by TEXT DEFAULT '',
    FOREIGN KEY(cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_actuations_cluster_id ON actuations(cluster_id);
CREATE INDEX IF NOT EXISTS idx_actuations_status ON actuations(status);

CREATE TABLE IF NOT EXISTS policies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    scope_type TEXT NOT NULL DEFAULT 'global',
    scope_value TEXT NOT NULL DEFAULT '',
    rule_type TEXT NOT NULL DEFAULT 'environment_restriction',
    rule_config TEXT NOT NULL DEFAULT '{}',
    severity TEXT NOT NULL DEFAULT 'medium',
    enabled INTEGER NOT NULL DEFAULT 1,
    fail_action TEXT NOT NULL DEFAULT 'block',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    actuation_id TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    steps_json TEXT NOT NULL DEFAULT '[]',
    required_approvers_json TEXT NOT NULL DEFAULT '[]',
    reason TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    decided_at TEXT,
    final_reason TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS alert_rules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    cluster_id TEXT NOT NULL,
    condition_type TEXT NOT NULL DEFAULT 'gpu_utilization',
    operator TEXT NOT NULL DEFAULT 'lt',
    threshold REAL NOT NULL DEFAULT 0.0,
    severity TEXT NOT NULL DEFAULT 'warning',
    enabled INTEGER NOT NULL DEFAULT 1,
    cooldown_minutes INTEGER NOT NULL DEFAULT 60,
    notification_channel_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(cluster_id) REFERENCES clusters(id)
);

CREATE TABLE IF NOT EXISTS alert_records (
    id TEXT PRIMARY KEY,
    rule_id TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    cluster_name TEXT DEFAULT '',
    severity TEXT NOT NULL DEFAULT 'warning',
    condition_type TEXT NOT NULL DEFAULT 'gpu_utilization',
    current_value REAL NOT NULL DEFAULT 0.0,
    threshold REAL NOT NULL DEFAULT 0.0,
    message TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'firing',
    triggered_at TEXT NOT NULL,
    resolved_at TEXT,
    acknowledged_by TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_alert_records_cluster ON alert_records(cluster_id);
CREATE INDEX IF NOT EXISTS idx_alert_records_status ON alert_records(status);

CREATE TABLE IF NOT EXISTS notification_channels (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    channel_type TEXT NOT NULL DEFAULT 'email',
    config_json TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notification_messages (
    id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    channel_name TEXT DEFAULT '',
    subject TEXT DEFAULT '',
    body TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    sent_at TEXT,
    error_message TEXT DEFAULT '',
    FOREIGN KEY(channel_id) REFERENCES notification_channels(id)
);

CREATE INDEX IF NOT EXISTS idx_notif_messages_channel ON notification_messages(channel_id);
"""

POSTGRES_INIT_SQL = """
CREATE TABLE IF NOT EXISTS clusters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    environment TEXT NOT NULL,
    connector_type TEXT NOT NULL,
    description TEXT,
    kube_context TEXT,
    kubeconfig_path TEXT,
    in_cluster INTEGER NOT NULL,
    credential_ref TEXT,
    options_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS check_reports (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES clusters(id),
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    overall_status TEXT NOT NULL,
    report_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reports_cluster_completed
ON check_reports(cluster_id, completed_at DESC);

CREATE TABLE IF NOT EXISTS cluster_state (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES clusters(id),
    collected_at TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    state_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_state_cluster_collected
ON cluster_state(cluster_id, collected_at DESC);

CREATE TABLE IF NOT EXISTS baselines (
    cluster_id TEXT PRIMARY KEY REFERENCES clusters(id),
    trace_id TEXT NOT NULL,
    set_at TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    node_count INTEGER NOT NULL DEFAULT 0,
    gpu_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES clusters(id),
    generated_at TEXT NOT NULL,
    analysis_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analyses_cluster_gen
ON analyses(cluster_id, generated_at DESC);

CREATE TABLE IF NOT EXISTS recommendations (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES clusters(id),
    generated_at TEXT NOT NULL,
    recs_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recs_cluster_gen
ON recommendations(cluster_id, generated_at DESC);

CREATE TABLE IF NOT EXISTS twins (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL UNIQUE REFERENCES clusters(id),
    synced_at TEXT NOT NULL,
    state_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS actuations (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    rec_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    dry_run INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    actions_json TEXT NOT NULL DEFAULT '[]',
    result_summary TEXT DEFAULT '',
    error_message TEXT DEFAULT '',
    rollback_of TEXT DEFAULT '',
    rolled_back_by TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_actuations_cluster_id ON actuations(cluster_id);
CREATE INDEX IF NOT EXISTS idx_actuations_status ON actuations(status);

CREATE TABLE IF NOT EXISTS policies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    scope_type TEXT NOT NULL DEFAULT 'global',
    scope_value TEXT NOT NULL DEFAULT '',
    rule_type TEXT NOT NULL DEFAULT 'environment_restriction',
    rule_config TEXT NOT NULL DEFAULT '{}',
    severity TEXT NOT NULL DEFAULT 'medium',
    enabled INTEGER NOT NULL DEFAULT 1,
    fail_action TEXT NOT NULL DEFAULT 'block',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    actuation_id TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    steps_json TEXT NOT NULL DEFAULT '[]',
    required_approvers_json TEXT NOT NULL DEFAULT '[]',
    reason TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    decided_at TEXT,
    final_reason TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS alert_rules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    cluster_id TEXT NOT NULL REFERENCES clusters(id),
    condition_type TEXT NOT NULL DEFAULT 'gpu_utilization',
    operator TEXT NOT NULL DEFAULT 'lt',
    threshold REAL NOT NULL DEFAULT 0.0,
    severity TEXT NOT NULL DEFAULT 'warning',
    enabled INTEGER NOT NULL DEFAULT 1,
    cooldown_minutes INTEGER NOT NULL DEFAULT 60,
    notification_channel_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alert_records (
    id TEXT PRIMARY KEY,
    rule_id TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    cluster_name TEXT DEFAULT '',
    severity TEXT NOT NULL DEFAULT 'warning',
    condition_type TEXT NOT NULL DEFAULT 'gpu_utilization',
    current_value REAL NOT NULL DEFAULT 0.0,
    threshold REAL NOT NULL DEFAULT 0.0,
    message TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'firing',
    triggered_at TEXT NOT NULL,
    resolved_at TEXT,
    acknowledged_by TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_alert_records_cluster ON alert_records(cluster_id);
CREATE INDEX IF NOT EXISTS idx_alert_records_status ON alert_records(status);

CREATE TABLE IF NOT EXISTS notification_channels (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    channel_type TEXT NOT NULL DEFAULT 'email',
    config_json TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notification_messages (
    id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL REFERENCES notification_channels(id),
    channel_name TEXT DEFAULT '',
    subject TEXT DEFAULT '',
    body TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    sent_at TEXT,
    error_message TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_notif_messages_channel ON notification_messages(channel_id);
"""



# ── Repository ──────────────────────────────────────────────────

class ClusterRepository:
    """Production-capable repository backed by SQLite or PostgreSQL.

    Use the factory function ``create_backend()`` or let the constructor
    resolve one from ``Settings`` automatically.
    """

    def __init__(
        self,
        database_path: Path | None = None,
        backend: DatabaseBackend | None = None,
    ) -> None:
        if backend is not None:
            self._backend = backend
        else:
            from .config import get_settings

            settings = get_settings()
            self._backend = create_backend(
                settings.resolved_database_url,
                database_path or settings.database_path,
                settings.database_pool_min,
                settings.database_pool_max,
            )
        self._initialize()

    def _initialize(self) -> None:
        self._backend.initialize()

    def _fmt(self, sql: str) -> tuple[str, type]:
        """Return (sql, row_type) adapted for the backend."""
        if isinstance(self._backend, PostgresBackend):
            pg_sql = sql.replace("?", "%s")
            pg_sql = pg_sql.replace(
                "INSERT OR REPLACE INTO",
                "INSERT INTO",
            )
            return pg_sql, dict
        return sql, sqlite3.Row

    def _fetchone(self, cursor: Any, row_type: type) -> Any | None:
        row = cursor.fetchone()
        if row is None:
            return None
        if row_type is dict:
            return dict(row)
        return row

    def _fetchall(self, cursor: Any, row_type: type) -> list:
        rows = cursor.fetchall()
        if row_type is dict:
            return [dict(r) for r in rows]
        return rows

    # ── Clusters ──────────────────────────────────────────────

    def create_cluster(self, payload: ClusterCreate) -> ClusterRecord:
        record = ClusterRecord(**payload.model_dump())
        sql, _ = self._fmt(
            "INSERT INTO clusters ("
            "id, name, environment, connector_type, description, kube_context, "
            "kubeconfig_path, in_cluster, credential_ref, region, options_json, "
            "created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        try:
            with self._backend.connect() as conn:
                conn.execute(sql, (
                    str(record.id), record.name, record.environment,
                    record.connector_type.value, record.description,
                    record.kube_context, record.kubeconfig_path,
                    int(record.in_cluster), record.credential_ref,
                    record.region, json.dumps(record.options),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ))
        except Exception as exc:
            raise RepositoryError(f"Cluster name already exists: {record.name}") from exc
        return record

    def upsert_cluster(self, payload: ClusterCreate) -> ClusterRecord:
        from .config import get_settings

        existing = self.get_cluster_by_name(payload.name)
        if existing is None:
            return self.create_cluster(payload)
        updated = existing.model_copy(update={
            **payload.model_dump(),
            "updated_at": datetime.now(timezone.utc),
        })
        sql, _ = self._fmt(
            "UPDATE clusters SET "
            "environment=?, connector_type=?, description=?, kube_context=?, "
            "kubeconfig_path=?, in_cluster=?, credential_ref=?, region=?, "
            "options_json=?, updated_at=? "
            "WHERE id=?"
        )
        with self._backend.connect() as conn:
            conn.execute(sql, (
                updated.environment, updated.connector_type.value,
                updated.description, updated.kube_context,
                updated.kubeconfig_path, int(updated.in_cluster),
                updated.credential_ref, updated.region,
                json.dumps(updated.options),
                updated.updated_at.isoformat(), str(updated.id),
            ))
        return updated

    def list_clusters(self) -> list[ClusterRecord]:
        sql, rt = self._fmt("SELECT * FROM clusters ORDER BY environment, name")
        with self._backend.connect() as conn:
            rows = self._fetchall(conn.execute(sql), rt)
        return [self._row_to_cluster(r) for r in rows]

    def get_cluster(self, cluster_id: UUID) -> ClusterRecord | None:
        sql, rt = self._fmt("SELECT * FROM clusters WHERE id=?")
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (str(cluster_id),)), rt)
        return self._row_to_cluster(row) if row else None

    def get_cluster_by_name(self, name: str) -> ClusterRecord | None:
        sql, rt = self._fmt("SELECT * FROM clusters WHERE name=?")
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (name,)), rt)
        return self._row_to_cluster(row) if row else None

    def delete_cluster(self, cluster_id: UUID) -> bool:
        child_sql = self._fmt("DELETE FROM check_reports WHERE cluster_id=?")[0]
        state_sql = self._fmt("DELETE FROM cluster_state WHERE cluster_id=?")[0]
        bl_sql = self._fmt("DELETE FROM baselines WHERE cluster_id=?")[0]
        an_sql = self._fmt("DELETE FROM analyses WHERE cluster_id=?")[0]
        rec_sql = self._fmt("DELETE FROM recommendations WHERE cluster_id=?")[0]
        twin_sql = self._fmt("DELETE FROM twins WHERE cluster_id=?")[0]
        act_sql = self._fmt("DELETE FROM actuations WHERE cluster_id=?")[0]
        main_sql = self._fmt("DELETE FROM clusters WHERE id=?")[0]

        cid = str(cluster_id)
        with self._backend.connect() as conn:
            conn.execute(child_sql, (cid,))
            conn.execute(state_sql, (cid,))
            conn.execute(bl_sql, (cid,))
            conn.execute(an_sql, (cid,))
            conn.execute(rec_sql, (cid,))
            conn.execute(twin_sql, (cid,))
            conn.execute(act_sql, (cid,))
            cursor = conn.execute(main_sql, (cid,))
            if isinstance(self._backend, PostgresBackend):
                return cursor.rowcount > 0
            return cursor.rowcount > 0

    # ── Checks ────────────────────────────────────────────────

    def save_report(self, report: EnvironmentCheckReport) -> None:
        sql, _ = self._fmt(
            "INSERT INTO check_reports ("
            "id, cluster_id, started_at, completed_at, overall_status, report_json"
            ") VALUES (?, ?, ?, ?, ?, ?)"
        )
        with self._backend.connect() as conn:
            conn.execute(sql, (
                str(report.id), str(report.cluster_id),
                report.started_at.isoformat(),
                report.completed_at.isoformat(),
                report.overall_status.value,
                report.model_dump_json(),
            ))

    def latest_report(self, cluster_id: UUID) -> EnvironmentCheckReport | None:
        sql, rt = self._fmt(
            "SELECT report_json FROM check_reports "
            "WHERE cluster_id=? ORDER BY completed_at DESC LIMIT 1"
        )
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (str(cluster_id),)), rt)
        return EnvironmentCheckReport.model_validate_json(row["report_json"]) if row else None

    def latest_reports(self) -> Iterable[tuple[ClusterRecord, EnvironmentCheckReport | None]]:
        for cluster in self.list_clusters():
            yield cluster, self.latest_report(cluster.id)

    # ── State / Telemetry ─────────────────────────────────────

    def _state_id(self, state: ClusterStateData) -> str:
        ts = state.collected_at.strftime("%Y%m%dT%H%M%S%f")
        return f"{state.cluster_id}_{ts}_{_new_uuid().hex[:8]}"

    def save_state(self, state: ClusterStateData) -> None:
        state_id = self._state_id(state)
        sql, _ = self._fmt(
            "INSERT INTO cluster_state (id, cluster_id, collected_at, generated_at, state_json) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        with self._backend.connect() as conn:
            conn.execute(sql, (
                state_id, str(state.cluster_id),
                state.collected_at.isoformat(),
                state.generated_at.isoformat(),
                state.model_dump_json(),
            ))

    def latest_state(self, cluster_id: UUID) -> ClusterStateData | None:
        sql, rt = self._fmt(
            "SELECT state_json FROM cluster_state "
            "WHERE cluster_id=? ORDER BY collected_at DESC LIMIT 1"
        )
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (str(cluster_id),)), rt)
        return ClusterStateData.model_validate_json(row["state_json"]) if row else None

    def list_state_summaries(self) -> list[tuple[ClusterRecord, ClusterStateData | None]]:
        result: list[tuple[ClusterRecord, ClusterStateData | None]] = []
        for cluster in self.list_clusters():
            result.append((cluster, self.latest_state(cluster.id)))
        return result

    def delete_state_for_cluster(self, cluster_id: UUID) -> None:
        sql, _ = self._fmt("DELETE FROM cluster_state WHERE cluster_id=?")
        with self._backend.connect() as conn:
            conn.execute(sql, (str(cluster_id),))

    # ── Traces ────────────────────────────────────────────────

    def list_traces(
        self, cluster_id: UUID, limit: int = 50, offset: int = 0
    ) -> list[tuple[str, ClusterStateData]]:
        sql, rt = self._fmt(
            "SELECT id, state_json FROM cluster_state "
            "WHERE cluster_id=? ORDER BY collected_at DESC LIMIT ? OFFSET ?"
        )
        with self._backend.connect() as conn:
            rows = self._fetchall(conn.execute(sql, (str(cluster_id), limit, offset)), rt)
        return [(r["id"], ClusterStateData.model_validate_json(r["state_json"])) for r in rows]

    def get_trace(self, cluster_id: UUID, trace_id: str) -> ClusterStateData | None:
        sql, rt = self._fmt(
            "SELECT state_json FROM cluster_state WHERE id=? AND cluster_id=?"
        )
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (trace_id, str(cluster_id))), rt)
        return ClusterStateData.model_validate_json(row["state_json"]) if row else None

    def trace_count(self, cluster_id: UUID) -> int:
        sql, rt = self._fmt("SELECT COUNT(*) AS cnt FROM cluster_state WHERE cluster_id=?")
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (str(cluster_id),)), rt)
        return row["cnt"] if row else 0

    # ── Baselines ─────────────────────────────────────────────

    def set_baseline(
        self, cluster_id: UUID, state: ClusterStateData, trace_id: str
    ) -> BaselineInfo:
        info = BaselineInfo(
            cluster_id=cluster_id,
            trace_id=trace_id,
            collected_at=state.collected_at,
            node_count=state.node_count,
            gpu_count=state.gpu_count,
        )
        if isinstance(self._backend, PostgresBackend):
            sql = (
                "INSERT INTO baselines (cluster_id, trace_id, set_at, collected_at, "
                "node_count, gpu_count) VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (cluster_id) DO UPDATE SET "
                "trace_id=EXCLUDED.trace_id, set_at=EXCLUDED.set_at, "
                "collected_at=EXCLUDED.collected_at, "
                "node_count=EXCLUDED.node_count, gpu_count=EXCLUDED.gpu_count"
            )
        else:
            sql = (
                "INSERT OR REPLACE INTO baselines "
                "(cluster_id, trace_id, set_at, collected_at, node_count, gpu_count) "
                "VALUES (?, ?, ?, ?, ?, ?)"
            )
        with self._backend.connect() as conn:
            conn.execute(sql, (
                str(cluster_id), trace_id, info.set_at.isoformat(),
                info.collected_at.isoformat(), info.node_count, info.gpu_count,
            ))
        return info

    def get_baseline(self, cluster_id: UUID) -> BaselineInfo | None:
        sql, rt = self._fmt("SELECT * FROM baselines WHERE cluster_id=?")
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (str(cluster_id),)), rt)
        if row is None:
            return None
        return BaselineInfo(
            cluster_id=UUID(row["cluster_id"]),
            trace_id=row["trace_id"],
            set_at=datetime.fromisoformat(row["set_at"]),
            collected_at=datetime.fromisoformat(row["collected_at"]),
            node_count=row["node_count"],
            gpu_count=row["gpu_count"],
        )

    def delete_baseline(self, cluster_id: UUID) -> None:
        sql, _ = self._fmt("DELETE FROM baselines WHERE cluster_id=?")
        with self._backend.connect() as conn:
            conn.execute(sql, (str(cluster_id),))

    # ── Analyses ──────────────────────────────────────────────

    def save_analysis(self, analysis: WorkloadAnalysisResult) -> None:
        if isinstance(self._backend, PostgresBackend):
            sql = (
                "INSERT INTO analyses (id, cluster_id, generated_at, analysis_json) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "cluster_id=EXCLUDED.cluster_id, generated_at=EXCLUDED.generated_at, "
                "analysis_json=EXCLUDED.analysis_json"
            )
        else:
            sql = (
                "INSERT OR REPLACE INTO analyses "
                "(id, cluster_id, generated_at, analysis_json) "
                "VALUES (?, ?, ?, ?)"
            )
        with self._backend.connect() as conn:
            conn.execute(sql, (
                str(analysis.id), str(analysis.cluster_id),
                analysis.generated_at.isoformat(),
                analysis.model_dump_json(),
            ))

    def latest_analysis(self, cluster_id: UUID) -> WorkloadAnalysisResult | None:
        sql, rt = self._fmt(
            "SELECT analysis_json FROM analyses "
            "WHERE cluster_id=? ORDER BY generated_at DESC LIMIT 1"
        )
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (str(cluster_id),)), rt)
        return WorkloadAnalysisResult.model_validate_json(row["analysis_json"]) if row else None

    def list_analyses(
        self, cluster_id: UUID, limit: int = 10
    ) -> list[WorkloadAnalysisResult]:
        sql, rt = self._fmt(
            "SELECT analysis_json FROM analyses "
            "WHERE cluster_id=? ORDER BY generated_at DESC LIMIT ?"
        )
        with self._backend.connect() as conn:
            rows = self._fetchall(conn.execute(sql, (str(cluster_id), limit)), rt)
        return [WorkloadAnalysisResult.model_validate_json(r["analysis_json"]) for r in rows]

    # ── Recommendations ───────────────────────────────────────

    def save_recommendations(self, recs: RecommendationSet) -> None:
        sql, _ = self._fmt(
            "INSERT INTO recommendations (id, cluster_id, generated_at, recs_json) "
            "VALUES (?, ?, ?, ?)"
        )
        with self._backend.connect() as conn:
            conn.execute(sql, (
                str(recs.id), str(recs.cluster_id),
                recs.generated_at.isoformat(), recs.model_dump_json(),
            ))

    def latest_recommendations(self, cluster_id: UUID) -> RecommendationSet | None:
        sql, rt = self._fmt(
            "SELECT recs_json FROM recommendations "
            "WHERE cluster_id=? ORDER BY generated_at DESC LIMIT 1"
        )
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (str(cluster_id),)), rt)
        return RecommendationSet.model_validate_json(row["recs_json"]) if row else None

    def list_recommendations(
        self, cluster_id: UUID, limit: int = 10
    ) -> list[RecommendationSet]:
        sql, rt = self._fmt(
            "SELECT recs_json FROM recommendations "
            "WHERE cluster_id=? ORDER BY generated_at DESC LIMIT ?"
        )
        with self._backend.connect() as conn:
            rows = self._fetchall(conn.execute(sql, (str(cluster_id), limit)), rt)
        return [RecommendationSet.model_validate_json(r["recs_json"]) for r in rows]

    # ── Digital Twin ──────────────────────────────────────────

    def save_twin(self, twin: TwinState) -> None:
        if isinstance(self._backend, PostgresBackend):
            sql = (
                "INSERT INTO twins (id, cluster_id, synced_at, state_json) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (cluster_id) DO UPDATE SET "
                "id=EXCLUDED.id, synced_at=EXCLUDED.synced_at, "
                "state_json=EXCLUDED.state_json"
            )
        else:
            sql = (
                "INSERT OR REPLACE INTO twins "
                "(id, cluster_id, synced_at, state_json) VALUES (?, ?, ?, ?)"
            )
        with self._backend.connect() as conn:
            conn.execute(sql, (
                str(twin.id), str(twin.cluster_id),
                twin.synced_at.isoformat(), twin.model_dump_json(),
            ))

    def get_twin(self, cluster_id: UUID) -> TwinState | None:
        sql, rt = self._fmt("SELECT state_json FROM twins WHERE cluster_id=?")
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (str(cluster_id),)), rt)
        return TwinState.model_validate_json(row["state_json"]) if row else None

    def delete_twin(self, cluster_id: UUID) -> None:
        sql, _ = self._fmt("DELETE FROM twins WHERE cluster_id=?")
        with self._backend.connect() as conn:
            conn.execute(sql, (str(cluster_id),))

    # ── Actuations ────────────────────────────────────────────

    def save_actuation(self, record: ActuationRecord) -> None:
        if isinstance(self._backend, PostgresBackend):
            sql = (
                "INSERT INTO actuations (id, cluster_id, rec_id, status, dry_run, "
                "started_at, completed_at, actions_json, result_summary, error_message, "
                "rollback_of, rolled_back_by) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "status=EXCLUDED.status, completed_at=EXCLUDED.completed_at, "
                "actions_json=EXCLUDED.actions_json, result_summary=EXCLUDED.result_summary, "
                "error_message=EXCLUDED.error_message"
            )
        else:
            sql = (
                "INSERT OR REPLACE INTO actuations (id, cluster_id, rec_id, status, dry_run, "
                "started_at, completed_at, actions_json, result_summary, error_message, "
                "rollback_of, rolled_back_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
        with self._backend.connect() as conn:
            conn.execute(sql, (
                str(record.id), str(record.cluster_id), str(record.rec_id),
                record.status.value, int(record.dry_run),
                record.started_at.isoformat(),
                record.completed_at.isoformat() if record.completed_at else None,
                record.model_dump_json(),
                record.result_summary, record.error_message,
                record.rollback_of, record.rolled_back_by,
            ))

    def get_actuation(
        self, cluster_id: UUID, actuation_id: UUID
    ) -> ActuationRecord | None:
        sql, rt = self._fmt(
            "SELECT actions_json FROM actuations WHERE cluster_id=? AND id=?"
        )
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (str(cluster_id), str(actuation_id))), rt)
        return ActuationRecord.model_validate_json(row["actions_json"]) if row else None

    def list_actuations(
        self, cluster_id: UUID, limit: int = 20
    ) -> list[ActuationRecord]:
        sql, rt = self._fmt(
            "SELECT actions_json FROM actuations "
            "WHERE cluster_id=? ORDER BY started_at DESC LIMIT ?"
        )
        with self._backend.connect() as conn:
            rows = self._fetchall(conn.execute(sql, (str(cluster_id), limit)), rt)
        return [ActuationRecord.model_validate_json(r["actions_json"]) for r in rows]

    # ── Policies ──────────────────────────────────────────────

    def save_policy(self, policy: PolicyRule) -> None:
        if isinstance(self._backend, PostgresBackend):
            sql = (
                "INSERT INTO policies (id, name, description, scope_type, scope_value, "
                "rule_type, rule_config, severity, enabled, fail_action, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "name=EXCLUDED.name, description=EXCLUDED.description, "
                "scope_type=EXCLUDED.scope_type, scope_value=EXCLUDED.scope_value, "
                "rule_type=EXCLUDED.rule_type, rule_config=EXCLUDED.rule_config, "
                "severity=EXCLUDED.severity, enabled=EXCLUDED.enabled, "
                "fail_action=EXCLUDED.fail_action, updated_at=EXCLUDED.updated_at"
            )
        else:
            sql = (
                "INSERT OR REPLACE INTO policies (id, name, description, scope_type, scope_value, "
                "rule_type, rule_config, severity, enabled, fail_action, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
        with self._backend.connect() as conn:
            conn.execute(sql, (
                str(policy.id), policy.name, policy.description,
                policy.scope_type, policy.scope_value,
                policy.rule_type, json.dumps(policy.rule_config),
                policy.severity.value, int(policy.enabled), policy.fail_action,
                policy.created_at.isoformat(), policy.updated_at.isoformat(),
            ))

    def get_policy(self, policy_id: UUID) -> PolicyRule | None:
        sql, rt = self._fmt("SELECT * FROM policies WHERE id=?")
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (str(policy_id),)), rt)
        if row is None:
            return None
        return self._row_to_policy(row)

    def list_policies(self) -> list[PolicyRule]:
        sql, rt = self._fmt("SELECT * FROM policies ORDER BY name")
        with self._backend.connect() as conn:
            rows = self._fetchall(conn.execute(sql), rt)
        return [self._row_to_policy(r) for r in rows]

    def delete_policy(self, policy_id: UUID) -> bool:
        sql, _ = self._fmt("DELETE FROM policies WHERE id=?")
        with self._backend.connect() as conn:
            cursor = conn.execute(sql, (str(policy_id),))
            return cursor.rowcount > 0

    # ── Approvals ─────────────────────────────────────────────

    def save_approval(self, record: ApprovalRecord) -> None:
        if isinstance(self._backend, PostgresBackend):
            sql = (
                "INSERT INTO approvals (id, actuation_id, cluster_id, status, "
                "steps_json, required_approvers_json, reason, created_at, decided_at, final_reason) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "status=EXCLUDED.status, steps_json=EXCLUDED.steps_json, "
                "decided_at=EXCLUDED.decided_at, final_reason=EXCLUDED.final_reason"
            )
        else:
            sql = (
                "INSERT OR REPLACE INTO approvals (id, actuation_id, cluster_id, status, "
                "steps_json, required_approvers_json, reason, created_at, decided_at, final_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
        import json as _json
        with self._backend.connect() as conn:
            conn.execute(sql, (
                str(record.id), str(record.actuation_id), str(record.cluster_id),
                record.status.value,
                _json.dumps([s.model_dump(mode="json") for s in record.steps]),
                _json.dumps(record.required_approvers),
                record.reason, record.created_at.isoformat(),
                record.decided_at.isoformat() if record.decided_at else None,
                record.final_reason,
            ))

    def get_approval(self, approval_id: UUID) -> ApprovalRecord | None:
        sql, rt = self._fmt("SELECT * FROM approvals WHERE id=?")
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (str(approval_id),)), rt)
        if row is None:
            return None
        return self._row_to_approval(row)

    def list_approvals(self, cluster_id: UUID | None = None) -> list[ApprovalRecord]:
        if cluster_id:
            sql, rt = self._fmt("SELECT * FROM approvals WHERE cluster_id=? ORDER BY created_at DESC")
            params = (str(cluster_id),)
        else:
            sql, rt = self._fmt("SELECT * FROM approvals ORDER BY created_at DESC")
            params = ()
        with self._backend.connect() as conn:
            rows = self._fetchall(conn.execute(sql, params), rt)
        return [self._row_to_approval(r) for r in rows]

    def list_approvals_for_actuation(self, actuation_id: UUID) -> list[ApprovalRecord]:
        sql, rt = self._fmt("SELECT * FROM approvals WHERE actuation_id=? ORDER BY created_at DESC")
        with self._backend.connect() as conn:
            rows = self._fetchall(conn.execute(sql, (str(actuation_id),)), rt)
        return [self._row_to_approval(r) for r in rows]

    def update_rec_status(
        self, cluster_id: UUID, rec_id: UUID, status: str, reason: str = ""
    ) -> ResourceRecommendation | None:
        sql, rt = self._fmt(
            "SELECT recs_json FROM recommendations "
            "WHERE cluster_id=? ORDER BY generated_at DESC LIMIT 1"
        )
        with self._backend.connect() as conn:
            row = self._fetchone(conn.execute(sql, (str(cluster_id),)), rt)
        if row is None:
            return None
        rec_set = RecommendationSet.model_validate_json(row["recs_json"])
        found = None
        for r in rec_set.recommendations:
            if r.id == rec_id:
                r.status = RecommendationStatus(status)
                found = r
                break
        if found is None:
            return None
        update_sql = self._fmt("UPDATE recommendations SET recs_json=? WHERE id=?")[0]
        with self._backend.connect() as conn:
            conn.execute(update_sql, (rec_set.model_dump_json(), str(rec_set.id)))
        return found

    # ── Alert Rules ───────────────────────────────────────────

    def save_alert_rule(self, rule: AlertRule) -> None:
        if isinstance(self._backend, PostgresBackend):
            sql = (
                "INSERT INTO alert_rules (id, name, description, cluster_id, condition_type, "
                "operator, threshold, severity, enabled, cooldown_minutes, "
                "notification_channel_ids, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "name=EXCLUDED.name, description=EXCLUDED.description, "
                "condition_type=EXCLUDED.condition_type, operator=EXCLUDED.operator, "
                "threshold=EXCLUDED.threshold, severity=EXCLUDED.severity, "
                "enabled=EXCLUDED.enabled, cooldown_minutes=EXCLUDED.cooldown_minutes, "
                "notification_channel_ids=EXCLUDED.notification_channel_ids, "
                "updated_at=EXCLUDED.updated_at"
            )
        else:
            sql = (
                "INSERT OR REPLACE INTO alert_rules (id, name, description, cluster_id, condition_type, "
                "operator, threshold, severity, enabled, cooldown_minutes, "
                "notification_channel_ids, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
        with self._backend.connect() as conn:
            conn.execute(sql, (
                str(rule.id), rule.name, rule.description, str(rule.cluster_id),
                rule.condition_type.value, rule.operator, rule.threshold,
                rule.severity.value, int(rule.enabled), rule.cooldown_minutes,
                json.dumps([str(c) for c in rule.notification_channel_ids]),
                rule.created_at.isoformat(), rule.updated_at.isoformat(),
            ))

    def delete_alert_rule(self, rule_id: UUID) -> bool:
        sql, _ = self._fmt("DELETE FROM alert_rules WHERE id=?")
        with self._backend.connect() as conn:
            return conn.execute(sql, (str(rule_id),)).rowcount > 0

    # ── Alert Records ─────────────────────────────────────────

    def save_alert_record(self, alert: AlertRecord) -> None:
        if isinstance(self._backend, PostgresBackend):
            sql = (
                "INSERT INTO alert_records (id, rule_id, cluster_id, cluster_name, severity, "
                "condition_type, current_value, threshold, message, status, "
                "triggered_at, resolved_at, acknowledged_by) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "status=EXCLUDED.status, resolved_at=EXCLUDED.resolved_at, "
                "acknowledged_by=EXCLUDED.acknowledged_by"
            )
        else:
            sql = (
                "INSERT OR REPLACE INTO alert_records (id, rule_id, cluster_id, cluster_name, severity, "
                "condition_type, current_value, threshold, message, status, "
                "triggered_at, resolved_at, acknowledged_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
        with self._backend.connect() as conn:
            conn.execute(sql, (
                str(alert.id), str(alert.rule_id), str(alert.cluster_id),
                alert.cluster_name, alert.severity.value, alert.condition_type.value,
                alert.current_value, alert.threshold, alert.message, alert.status,
                alert.triggered_at.isoformat(),
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                alert.acknowledged_by,
            ))

    # ── Notification Channels ─────────────────────────────────

    def save_notification_channel(self, channel: NotificationChannel) -> None:
        if isinstance(self._backend, PostgresBackend):
            sql = (
                "INSERT INTO notification_channels (id, name, channel_type, config_json, enabled, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "name=EXCLUDED.name, channel_type=EXCLUDED.channel_type, "
                "config_json=EXCLUDED.config_json, enabled=EXCLUDED.enabled"
            )
        else:
            sql = (
                "INSERT OR REPLACE INTO notification_channels (id, name, channel_type, config_json, enabled, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)"
            )
        with self._backend.connect() as conn:
            conn.execute(sql, (
                str(channel.id), channel.name, channel.channel_type.value,
                json.dumps(channel.config), int(channel.enabled),
                channel.created_at.isoformat(),
            ))

    def delete_notification_channel(self, channel_id: UUID) -> bool:
        sql, _ = self._fmt("DELETE FROM notification_channels WHERE id=?")
        with self._backend.connect() as conn:
            return conn.execute(sql, (str(channel_id),)).rowcount > 0

    def list_notification_channels(self) -> list[NotificationChannel]:
        sql, rt = self._fmt("SELECT * FROM notification_channels ORDER BY name")
        with self._backend.connect() as conn:
            rows = self._fetchall(conn.execute(sql), rt)
        return [self._row_to_notification_channel(r) for r in rows]

    # ── Notification Messages ─────────────────────────────────

    def save_notification_message(self, msg: NotificationMessage) -> None:
        if isinstance(self._backend, PostgresBackend):
            sql = (
                "INSERT INTO notification_messages (id, channel_id, channel_name, subject, body, status, sent_at, error_message) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "status=EXCLUDED.status, sent_at=EXCLUDED.sent_at, "
                "error_message=EXCLUDED.error_message"
            )
        else:
            sql = (
                "INSERT OR REPLACE INTO notification_messages (id, channel_id, channel_name, subject, body, status, sent_at, error_message) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            )
        with self._backend.connect() as conn:
            conn.execute(sql, (
                str(msg.id), str(msg.channel_id), msg.channel_name,
                msg.subject, msg.body, msg.status,
                msg.sent_at.isoformat() if msg.sent_at else None,
                msg.error_message,
            ))

    def list_notification_messages(self, channel_id: UUID | None = None) -> list[NotificationMessage]:
        if channel_id:
            sql, rt = self._fmt("SELECT * FROM notification_messages WHERE channel_id=? ORDER BY sent_at DESC")
            params = (str(channel_id),)
        else:
            sql, rt = self._fmt("SELECT * FROM notification_messages ORDER BY sent_at DESC")
            params = ()
        with self._backend.connect() as conn:
            rows = self._fetchall(conn.execute(sql, params), rt)
        return [self._row_to_notification_message(r) for r in rows]

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _row_to_policy(row: Any) -> PolicyRule:
        return PolicyRule(
            id=UUID(row["id"]),
            name=row["name"],
            description=row["description"],
            scope_type=row["scope_type"],
            scope_value=row["scope_value"],
            rule_type=row["rule_type"],
            rule_config=json.loads(row["rule_config"]),
            severity=row["severity"],
            enabled=bool(row["enabled"]),
            fail_action=row["fail_action"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_approval(row: Any) -> ApprovalRecord:
        import json as _json
        return ApprovalRecord(
            id=UUID(row["id"]),
            actuation_id=UUID(row["actuation_id"]),
            cluster_id=UUID(row["cluster_id"]),
            status=row["status"],
            steps=[ApprovalStep.model_validate(s) for s in _json.loads(row["steps_json"])],
            required_approvers=_json.loads(row["required_approvers_json"]),
            reason=row["reason"],
            created_at=datetime.fromisoformat(row["created_at"]),
            decided_at=datetime.fromisoformat(row["decided_at"]) if row["decided_at"] else None,
            final_reason=row["final_reason"],
        )

    @staticmethod
    def _row_to_cluster(row: Any) -> ClusterRecord:
        d = dict(row)
        return ClusterRecord(
            id=UUID(d["id"]),
            name=d["name"],
            environment=d["environment"],
            connector_type=d["connector_type"],
            description=d["description"],
            kube_context=d["kube_context"],
            kubeconfig_path=d["kubeconfig_path"],
            in_cluster=bool(d["in_cluster"]),
            credential_ref=d["credential_ref"],
            region=d.get("region"),
            options=json.loads(d["options_json"]),
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
        )

    @staticmethod
    def _row_to_notification_channel(row: Any) -> NotificationChannel:
        from .schemas import NotificationChannelType
        return NotificationChannel(
            id=UUID(row["id"]),
            name=row["name"],
            channel_type=NotificationChannelType(row["channel_type"]),
            config=json.loads(row["config_json"]),
            enabled=bool(row["enabled"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_notification_message(row: Any) -> NotificationMessage:
        return NotificationMessage(
            id=UUID(row["id"]),
            channel_id=UUID(row["channel_id"]),
            channel_name=row["channel_name"],
            subject=row["subject"],
            body=row["body"],
            status=row["status"],
            sent_at=datetime.fromisoformat(row["sent_at"]) if row["sent_at"] else None,
            error_message=row["error_message"],
        )

    def close(self) -> None:
        if isinstance(self._backend, PostgresBackend):
            self._backend.close()
