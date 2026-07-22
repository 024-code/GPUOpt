"""add_domain_events_table

Revision ID: e3a5f7b9241c
Revises: 925f4ea7246b
Create Date: 2026-07-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


revision: str = 'e3a5f7b9241c'
down_revision: Union[str, None] = '925f4ea7246b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS domain_events (
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
);

CREATE INDEX IF NOT EXISTS idx_domain_events_domain_ts
ON domain_events(domain, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_domain_events_domain_cluster
ON domain_events(domain, cluster_id);

CREATE INDEX IF NOT EXISTS idx_domain_events_domain_job
ON domain_events(domain, job_id);
"""

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS domain_events (
    id BIGSERIAL PRIMARY KEY,
    domain TEXT NOT NULL,
    payload TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    cluster_id TEXT,
    node_id TEXT,
    job_id TEXT,
    tenant_id TEXT,
    model_id TEXT,
    event_type TEXT
);

CREATE INDEX IF NOT EXISTS idx_domain_events_domain_ts
ON domain_events(domain, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_domain_events_domain_cluster
ON domain_events(domain, cluster_id);

CREATE INDEX IF NOT EXISTS idx_domain_events_domain_job
ON domain_events(domain, job_id);
"""


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
        op.execute("DROP TABLE IF EXISTS domain_events CASCADE;")
    else:
        op.execute("PRAGMA foreign_keys = OFF;")
        op.execute("DROP TABLE IF EXISTS domain_events;")
        op.execute("PRAGMA foreign_keys = ON;")
