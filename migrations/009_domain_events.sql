CREATE TABLE IF NOT EXISTS domain_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    domain      TEXT NOT NULL,
    payload     TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    cluster_id  TEXT,
    node_id     TEXT,
    job_id      TEXT,
    tenant_id   TEXT,
    model_id    TEXT,
    event_type  TEXT
);

CREATE INDEX IF NOT EXISTS idx_domain_events_domain_ts
    ON domain_events(domain, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_domain_events_domain_cluster
    ON domain_events(domain, cluster_id);

CREATE INDEX IF NOT EXISTS idx_domain_events_domain_job
    ON domain_events(domain, job_id);
