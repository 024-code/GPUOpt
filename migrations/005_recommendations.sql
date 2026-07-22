-- Schema migration for R0.5: Recommendation MVP.
-- Adds the recommendations table for storing generated recommendation sets.

CREATE TABLE IF NOT EXISTS recommendations (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    recs_json TEXT NOT NULL,
    FOREIGN KEY(cluster_id) REFERENCES clusters(id)
);

CREATE INDEX IF NOT EXISTS idx_recs_cluster_gen
ON recommendations(cluster_id, generated_at DESC);
