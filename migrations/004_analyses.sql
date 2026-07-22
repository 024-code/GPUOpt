-- Schema migration for R0.4: Workload Analysis & Resource Profiling.
-- Adds the analyses table for storing workload analysis results.

CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    FOREIGN KEY(cluster_id) REFERENCES clusters(id)
);

CREATE INDEX IF NOT EXISTS idx_analyses_cluster_gen
ON analyses(cluster_id, generated_at DESC);
