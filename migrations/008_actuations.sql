CREATE TABLE IF NOT EXISTS actuations (
    id          TEXT PRIMARY KEY,
    cluster_id  TEXT NOT NULL,
    rec_id      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    dry_run     INTEGER NOT NULL DEFAULT 0,
    started_at  TEXT NOT NULL,
    completed_at TEXT,
    actions_json TEXT NOT NULL DEFAULT '[]',
    result_summary TEXT DEFAULT '',
    error_message TEXT DEFAULT '',
    rollback_of TEXT DEFAULT '',
    rolled_back_by TEXT DEFAULT '',
    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_actuations_cluster_id ON actuations(cluster_id);
CREATE INDEX IF NOT EXISTS idx_actuations_status ON actuations(status);
