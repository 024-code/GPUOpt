-- Schema migration for R0.7: Digital Twin Service.
-- Adds the twins table for storing digital twin state snapshots.

CREATE TABLE IF NOT EXISTS twins (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL UNIQUE,
    synced_at TEXT NOT NULL,
    state_json TEXT NOT NULL,
    FOREIGN KEY(cluster_id) REFERENCES clusters(id)
);
