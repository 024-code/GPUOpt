-- Schema migration for R0.3: Trace replay & baseline simulation.
-- Adds the baselines table for marking a state snapshot as the comparison baseline.

CREATE TABLE IF NOT EXISTS baselines (
    cluster_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    set_at TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    node_count INTEGER NOT NULL DEFAULT 0,
    gpu_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(cluster_id) REFERENCES clusters(id)
);
