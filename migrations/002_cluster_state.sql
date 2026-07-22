-- Schema migration for R0.2: Telemetry normalization & cluster state.
-- Adds the cluster_state table for persisting telemetry and state snapshots.

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
