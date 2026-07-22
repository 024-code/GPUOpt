# API Reference

Base URL: `http://127.0.0.1:8080`

Interactive documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## Health Endpoints

### Liveness Probe

```
GET /health/live
```

Returns a simple alive signal. Use this for Kubernetes liveness probes.

**Response** `200 OK`

```json
{
  "status": "alive"
}
```

### Readiness Probe

```
GET /health/ready
```

Verifies the database is accessible and returns the number of registered clusters. Use this for Kubernetes readiness probes.

**Response** `200 OK`

```json
{
  "status": "ready",
  "registered_clusters": 3
}
```

### Detailed Health

```
GET /health/detailed
```

Provides system-level and cluster-level health information.

**Response** `200 OK`

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "system": {
    "python_version": "3.11.9 | packaged by ...",
    "platform": "Linux-...",
    "architecture": "x86_64"
  },
  "configuration": {
    "environment": "sandbox",
    "database_path": "./data/gpuopt.db",
    "allow_mock_gpu": true
  },
  "clusters": {
    "total": 3,
    "by_environment": {"sandbox": 1, "staging": 1, "production": 1},
    "by_connector_type": {"mock": 1, "kubernetes": 2}
  }
}
```

---

## System Info

### System Information

```
GET /api/v1/info
```

Returns metadata about the GPUOpt instance.

**Response** `200 OK`

```json
{
  "name": "GPUOpt Backend Sandbox",
  "version": "0.1.0",
  "environment": "development",
  "documentation": "/docs",
  "health": "/health/detailed",
  "metrics": "/metrics"
}
```

---

## Metrics

### Prometheus Metrics

```
GET /metrics
```

Exposes Prometheus-format metrics. Not included in OpenAPI schema.

**Response** `200 OK` with `Content-Type: text/plain; version=0.0.4`

**Exposed metrics:**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `gpuopt_environment_check_runs_total` | Counter | `cluster`, `environment`, `status` | Total check runs |
| `gpuopt_environment_check_duration_seconds` | Histogram | `cluster`, `environment` | Check execution duration |
| `gpuopt_cluster_health_status` | Gauge | `cluster`, `environment` | 0=unchecked, 1=healthy, 2=warning, 3=failing |

---

## Clusters

### Register a Cluster

```
POST /api/v1/clusters
```

**Request Body** `application/json`

```json
{
  "name": "staging-gpu",
  "environment": "staging",
  "connector_type": "kubernetes",
  "description": "Real NVIDIA GPU staging cluster",
  "kube_context": "gpuopt-staging",
  "kubeconfig_path": "~/.kube/config",
  "in_cluster": false,
  "credential_ref": null,
  "options": {
    "allow_mock_gpu": false
  }
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | Yes | - | Unique name (2-120 chars) |
| `environment` | string | No | `"development"` | Environment label (max 40 chars) |
| `connector_type` | string | Yes | - | `"mock"` or `"kubernetes"` |
| `description` | string | No | `null` | Free-text description (max 500 chars) |
| `kube_context` | string | No | `null` | Kubeconfig context name |
| `kubeconfig_path` | string | No | `null` | Path to kubeconfig file |
| `in_cluster` | boolean | No | `false` | Use in-cluster ServiceAccount |
| `credential_ref` | string | No | `null` | Reference to stored credentials |
| `options` | object | No | `{}` | Connector-specific options |

**Validation rules:**
- `kubeconfig_path` must not contain inline kubeconfig data (rejects `apiVersion:` in the value).
- Cluster `name` must be unique (409 Conflict on duplicate).

**Response** `201 Created`

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "staging-gpu",
  "environment": "staging",
  "connector_type": "kubernetes",
  "description": "Real NVIDIA GPU staging cluster",
  "kube_context": "gpuopt-staging",
  "kubeconfig_path": "~/.kube/config",
  "in_cluster": false,
  "credential_ref": null,
  "options": {
    "allow_mock_gpu": false
  },
  "created_at": "2025-01-15T10:30:00+00:00",
  "updated_at": "2025-01-15T10:30:00+00:00"
}
```

**Errors:**

| Status | Condition |
|--------|-----------|
| `409` | Cluster name already exists |
| `422` | Validation error (missing fields, invalid values) |

---

### Upsert a Cluster by Name

```
PUT /api/v1/clusters/by-name/{name}
```

Creates or updates a cluster by name. The `name` in the URL path must match `name` in the request body.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Cluster name to upsert |

**Request Body:** Same as `POST /api/v1/clusters`.

**Response** `200 OK` — Returns the `ClusterRecord` (same shape as POST).

**Errors:**

| Status | Condition |
|--------|-----------|
| `400` | Path name does not match payload name |

---

### List All Clusters

```
GET /api/v1/clusters
```

Returns all registered clusters ordered by environment then name.

**Response** `200 OK`

```json
[
  {
    "id": "...",
    "name": "local-mock",
    "environment": "sandbox",
    "connector_type": "mock",
    ...
  }
]
```

---

### Get a Cluster

```
GET /api/v1/clusters/{cluster_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `cluster_id` | UUID | Cluster ID |

**Response** `200 OK` — Single `ClusterRecord`.

**Errors:**

| Status | Condition |
|--------|-----------|
| `404` | Cluster not found |

---

### Delete a Cluster

```
DELETE /api/v1/clusters/{cluster_id}
```

Deletes the cluster and all associated check reports.

**Response** `204 No Content`

**Errors:**

| Status | Condition |
|--------|-----------|
| `404` | Cluster not found |

---

## Environment Checks

### Run Checks on a Single Cluster

```
POST /api/v1/clusters/{cluster_id}/checks
```

Executes all readiness checks against the specified cluster. The connector is built based on the cluster's `connector_type`.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `cluster_id` | UUID | Cluster ID |

**Response** `200 OK`

```json
{
  "id": "report-uuid",
  "cluster_id": "cluster-uuid",
  "cluster_name": "local-mock",
  "environment": "sandbox",
  "started_at": "2025-01-15T10:30:00+00:00",
  "completed_at": "2025-01-15T10:30:01+00:00",
  "overall_status": "pass",
  "checks": [
    {
      "name": "api_server",
      "status": "pass",
      "message": "API Server is available.",
      "latency_ms": 12.5,
      "details": {"ready": true, "version": "mock-v1"},
      "remediation": null
    },
    {
      "name": "node_inventory",
      "status": "pass",
      "message": "Discovered 3 node(s); 3 Ready.",
      "latency_ms": 5.2,
      "details": {"nodes": [...]},
      "remediation": null
    }
  ],
  "summary": {
    "pass": 7,
    "warn": 0,
    "fail": 0,
    "skip": 0
  }
}
```

**Overall status logic:**
- `pass` — all checks pass
- `warn` — any check warns (optional component missing, mock GPU)
- `fail` — any check fails (API unreachable, RBAC missing, no GPUs)
- `skip` — all checks skipped

**Errors:**

| Status | Condition |
|--------|-----------|
| `404` | Cluster not found |

---

### Get Latest Check Report

```
GET /api/v1/clusters/{cluster_id}/checks/latest
```

Returns the most recent check report for the given cluster.

**Response** `200 OK` — `EnvironmentCheckReport` (same shape as above).

**Errors:**

| Status | Condition |
|--------|-----------|
| `404` | Cluster not found or no reports exist |

---

### Run Checks on All Clusters

```
POST /api/v1/environments/check-all
```

Iterates all registered clusters and runs readiness checks on each. Returns an array of reports.

**Response** `200 OK`

```json
[
  { "cluster_name": "local-mock", "overall_status": "pass", ... },
  { "cluster_name": "staging-gpu", "overall_status": "warn", ... }
]
```

---

### Environment Health Summary

```
GET /api/v1/environments/summary
```

Aggregated health summary across all registered clusters and environments.

**Response** `200 OK`

```json
{
  "generated_at": "2025-01-15T10:35:00+00:00",
  "clusters": 3,
  "healthy": 2,
  "warning": 1,
  "failing": 0,
  "unchecked": 0,
  "environments": {
    "sandbox": {
      "clusters": 1,
      "healthy": 1,
      "warning": 0,
      "failing": 0,
      "unchecked": 0
    },
    "staging": {
      "clusters": 1,
      "healthy": 1,
      "warning": 0,
      "failing": 0,
      "unchecked": 0
    },
    "production": {
      "clusters": 1,
      "healthy": 0,
      "warning": 1,
      "failing": 0,
      "unchecked": 0
    }
  }
}
```

---

## Check Items Performed

Each `POST .../checks` execution runs these checks via the connector:

| Check Name | Required | Description |
|------------|----------|-------------|
| `api_server` | Yes | Kubernetes API server reachability and version |
| `rbac_permissions` | Yes | Read-only RBAC: list nodes, pods, CRDs |
| `node_inventory` | Yes | Node count and Ready status |
| `gpu_inventory` | Yes | `nvidia.com/gpu` extended resources or mock labels |
| `gpu_operator` | No | NVIDIA GPU Operator pods |
| `dcgm_exporter` | Yes | DCGM exporter pods and metrics endpoint |
| `batch_scheduler` | No | Kueue or Volcano CRD presence |
| `prometheus` | Yes | Prometheus service discovery |

Each check returns `status`, `message`, optional `latency_ms`, `details`, and `remediation` guidance.
