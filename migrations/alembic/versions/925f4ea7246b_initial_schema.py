"""initial_schema

Revision ID: 925f4ea7246b
Revises: 
Create Date: 2026-07-21 23:34:51.477825

"""
from typing import Sequence, Union

from alembic import op


revision: str = '925f4ea7246b'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


SQLITE_SCHEMA = """
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
"""

POSTGRES_SCHEMA = """
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
"""

TABLES = [
    "actuations",
    "twins",
    "recommendations",
    "analyses",
    "baselines",
    "cluster_state",
    "check_reports",
    "clusters",
]


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if _is_postgres():
        for stmt in POSTGRES_SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                op.execute(stmt + ";")
    else:
        op.execute("PRAGMA foreign_keys = OFF;")
        for stmt in SQLITE_SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                op.execute(stmt + ";")
        op.execute("PRAGMA foreign_keys = ON;")


def downgrade() -> None:
    if _is_postgres():
        for table in TABLES:
            op.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
    else:
        op.execute("PRAGMA foreign_keys = OFF;")
        for table in TABLES:
            op.execute(f"DROP TABLE IF EXISTS {table};")
        op.execute("PRAGMA foreign_keys = ON;")
