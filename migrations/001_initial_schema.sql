-- Initial schema for GPUOpt Backend Sandbox.
-- Matches the schema created by repository.py at startup.

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
