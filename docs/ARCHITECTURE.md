# GPUOpt Backend Architecture

## 1. System Overview

GPUOpt is a layered FastAPI application for registering Kubernetes clusters and assessing their GPU platform readiness. It follows a **clean architecture** pattern with strict separation of concerns: API layer, service layer, repository layer, and connector abstraction.

### Context & Scope

```
                    +--------------------------+
                    |    External Consumers     |
                    | (Dashboard / CI / CLI)    |
                    +------------+-------------+
                                 |
                    HTTP JSON    |    Prometheus scrape
                                 |
                    +------------v-------------+
                    |      GPUOpt Backend       |
                    |  (FastAPI + Uvicorn)      |
                    +---+----+----+----+----+---+
                        |    |    |    |    |
              SQLite    |    |    |    |    +---> Prometheus Metrics
              (sandbox) |    |    |    |
                        |    |    |    +--------> Mock JSON snapshots
                        |    |    |
                        |    |    +-------------> Kubernetes API (read-only)
                        |    |
                        |    +------------------> Prometheus / DCGM exporters
                        |
                        +-----------------------> PostgreSQL (production)
```

### Design Tenets

| Principle | Rationale |
|-----------|-----------|
| **Read-only by default** | All K8s interactions are non-mutating. No mutation permissions until the full actuation pipeline is hardened. |
| **Connector abstraction** | Different backends (mock, K8s, future cloud APIs) share the same check interface. |
| **Stateless API, stateful repository** | API handlers carry no mutable state; all state lives in the repository. |
| **Singleton DI via lru_cache** | Repository and service instances are created once and cached. Test fixtures clear the cache for isolation. |
| **Middleware pipeline** | Cross-cutting concerns (logging, auth, rate limiting, CORS) are middleware, not mixed into business logic. |
| **Defers** | Heavy imports (Kubernetes client) are deferred so mock mode never requires K8s dependencies. |

---

## 2. Layer Architecture

```
+------------------------------------------------------------------+
|                        Middleware Pipeline                         |
|  CORSMiddleware -> RateLimitMiddleware -> APIKeyAuthMiddleware     |
|  -> RequestLoggingMiddleware                                       |
+------------------------------------------------------------------+
                            |
                            v
+------------------------------------------------------------------+
|                     API Layer (api.py)                             |
|  Route handlers | Request validation | Response serialization      |
|  GET /health/*  | POST/PUT/DELETE /api/v1/clusters/*              |
|  Dependency injection via Depends(get_repository / get_check_...) |
+------------------------------------------------------------------+
                            |
                            v
+------------------------------------------------------------------+
|                   Service Layer (services.py)                      |
|  EnvironmentCheckService                                          |
|    - check_cluster()   - check_all()   - summarize()              |
|  Prometheus metrics: CHECK_RUNS, CHECK_DURATION, CLUSTER_STATUS    |
|  Orchestrates connector.build -> run_checks -> save_report         |
+------------------------------------------------------------------+
                            |
                    +-------+-------+
                    |               |
                    v               v
+-----------------------------+  +-------------------------------+
| Repository Layer            |  | Connector System               |
| (repository.py)             |  | (connectors/)                  |
|                             |  |                                |
| SQLite CRUD via RLock       |  | base.py  (abstract interface)  |
| clusters table              |  | factory.py (routing)           |
| check_reports table         |  | mock.py   (JSON snapshots)     |
| Serializes options as JSON  |  | kubernetes.py (real K8s API)   |
+-----------------------------+  +-------------------------------+
```

### 2.1 Middleware Pipeline (order of execution)

The pipeline is constructed via `app.add_middleware()`. In Starlette, the **last added middleware executes first** (outermost wraps innermost):

```
Request IN
    |
    v
CORSMiddleware (outermost: handles OPTIONS preflight, adds CORS headers)
    |
    v
RateLimitMiddleware (checks per-IP limits, adds X-RateLimit-* headers)
    |
    v
APIKeyAuthMiddleware (validates X-API-Key if configured, skips public paths)
    |
    v
RequestLoggingMiddleware (generates correlation ID, logs start/end, adds X-Correlation-ID, X-Response-Time)
    |
    v
FastAPI Router -> Route Handler -> Business Logic
    |
    v
Response flows back up through each middleware layer
```

**Public paths** (`/health/live`, `/health/ready`, `/metrics`, `/docs`, `/openapi.json`, `/redoc`) bypass API key auth but still pass through rate limiting (the health/metrics paths are also excluded from rate limiting).

### 2.2 API Layer (`api.py`)

Each route handler is a thin function that:
1. Declares dependencies via type-annotated `Depends()` parameters.
2. Calls the service or repository method.
3. Maps business exceptions to HTTP status codes.

**Route groups:**

| Tag | Prefix | Handlers |
|-----|--------|----------|
| `health` | `/health/` | `liveness`, `readiness`, `detailed_health` |
| `system` | `/api/v1/info` | `system_info` |
| `clusters` | `/api/v1/clusters` | `create_cluster`, `upsert_cluster`, `list_clusters`, `get_cluster`, `delete_cluster` |
| `environment checks` | `/api/v1/clusters/{id}/checks`, `/api/v1/environments/` | `run_cluster_check`, `latest_cluster_check`, `check_all_environments`, `environment_summary` |
| (none) | `/metrics` | `metrics` |

**Error mapping:**

| Exception | HTTP Status |
|-----------|-------------|
| `KeyError` from service | 404 Not Found |
| `RepositoryError` (duplicate name) | 409 Conflict |
| Pydantic `ValidationError` | 422 Unprocessable Entity |
| `GPUOptException` hierarchy | 400/404/409/500 (per exception) |
| Unhandled `Exception` | 500 Internal Server Error |

### 2.3 Service Layer (`services.py`)

`EnvironmentCheckService` is the orchestrator. Its lifecycle:

```
check_cluster(cluster_id)
  |
  |-- repository.get_cluster(cluster_id)      # Load cluster record
  |-- build_connector(cluster)                 # Factory: MockConnector | KubernetesConnector
  |-- connector.run_checks()                    # Execute all checks
  |-- _overall_status(checks)                  # FAIL > WARN > SKIP > PASS
  |-- Build EnvironmentCheckReport
  |-- repository.save_report(report)            # Persist
  |-- UPDATE CHECK_RUNS (Counter)              # Prometheus metric
  |-- UPDATE CLUSTER_STATUS (Gauge)            # Prometheus metric
  |-- RETURN report
```

**Prometheus metrics emitted:**

| Metric | Type | Labels | When |
|--------|------|--------|------|
| `gpuopt_environment_check_runs_total` | Counter | cluster, environment, status | After each check run |
| `gpuopt_environment_check_duration_seconds` | Histogram | cluster, environment | During check execution (context manager) |
| `gpuopt_cluster_health_status` | Gauge | cluster, environment | After each check run |

**Status priority cascade:**

```
If ANY check is FAIL   -> overall FAIL
Elif ANY check is WARN -> overall WARN
Elif ALL checks are SKIP -> overall SKIP
Else                   -> overall PASS
```

### 2.4 Repository Layer (`repository.py`)

`ClusterRepository` provides thread-safe SQLite persistence:

**Concurrency model:**
- `threading.RLock` protects all read/write operations.
- `check_same_thread=False` allows the connection to be shared across async threads in Uvicorn.
- Each method acquires the lock, creates a fresh connection (context manager), executes, and closes.

**Tables:**

```sql
clusters (
    id TEXT PRIMARY KEY,          -- UUID string
    name TEXT NOT NULL UNIQUE,     -- Human-readable, unique
    environment TEXT NOT NULL,     -- sandbox|development|staging|production
    connector_type TEXT NOT NULL,  -- mock|kubernetes
    description TEXT,
    kube_context TEXT,
    kubeconfig_path TEXT,
    in_cluster INTEGER NOT NULL,   -- boolean
    credential_ref TEXT,
    options_json TEXT NOT NULL,    -- JSON blob
    created_at TEXT NOT NULL,      -- ISO 8601
    updated_at TEXT NOT NULL       -- ISO 8601
)

check_reports (
    id TEXT PRIMARY KEY,          -- UUID string
    cluster_id TEXT NOT NULL,      -- FK -> clusters.id
    started_at TEXT NOT NULL,      -- ISO 8601
    completed_at TEXT NOT NULL,    -- ISO 8601
    overall_status TEXT NOT NULL,  -- pass|warn|fail|skip
    report_json TEXT NOT NULL      -- Full report as JSON
)

INDEX idx_reports_cluster_completed ON check_reports(cluster_id, completed_at DESC)
```

**CRUD surface:**

| Method | SQL | Notes |
|--------|-----|-------|
| `create_cluster` | INSERT | Raises `RepositoryError` on name conflict |
| `upsert_cluster` | INSERT or UPDATE | Looks up by name; updates if exists |
| `get_cluster` | SELECT by UUID | Returns `ClusterRecord` or `None` |
| `get_cluster_by_name` | SELECT by name | Returns `ClusterRecord` or `None` |
| `list_clusters` | SELECT all ORDER BY environment, name | Returns `list[ClusterRecord]` |
| `delete_cluster` | DELETE (reports then cluster) | Cascading delete |
| `save_report` | INSERT | Stores full report JSON |
| `latest_report` | SELECT ... ORDER BY completed_at DESC LIMIT 1 | Returns `EnvironmentCheckReport` or `None` |
| `latest_reports` | Generator over `list_clusters()` + `latest_report()` | Yields `(ClusterRecord, Report | None)` |

### 2.5 Dependency Injection (`dependencies.py`)

```python
get_repository()    -> ClusterRepository     # Singleton via @lru_cache(maxsize=1)
get_check_service() -> EnvironmentCheckService  # Singleton via @lru_cache(maxsize=1)
```

The `lru_cache` ensures a single instance across all requests. Test fixtures call `cache_clear()` on all three cached functions before and after each test to provide database isolation.

### 2.6 Connector System (`connectors/`)

#### Abstract Interface (`base.py`)

```python
class ClusterConnector(ABC):
    def __init__(self, cluster: ClusterRecord):
        self.cluster = cluster

    @abstractmethod
    def run_checks(self) -> list[CheckItem]:
        """Run read-only checks, return individual results."""
```

Every connector receives the `ClusterRecord` at construction time, giving it access to the cluster's environment, options, and authentication context.

#### Factory (`factory.py`)

```python
def build_connector(cluster: ClusterRecord) -> ClusterConnector:
    if cluster.connector_type == ConnectorType.MOCK:
        return MockConnector(cluster)
    if cluster.connector_type == ConnectorType.KUBERNETES:
        return KubernetesConnector(cluster)
    raise ValueError(...)
```

Adding a new connector type requires:
1. A new subclass of `ClusterConnector`.
2. A new `ConnectorType` enum value.
3. A new branch in `build_connector()`.

#### Mock Connector (`mock.py`)

Used for local development, CI, and sandbox environments without real GPU hardware. The mock connector:

1. Reads a JSON snapshot file from `cluster.options["snapshot_path"]`, or falls back to a built-in default.
2. Executes 8 synthetic checks against the snapshot data.
3. The snapshot format simulates a healthy K8s cluster with GPU nodes.

**Mock checks executed:**

| Check | Source | Required |
|-------|--------|----------|
| `api_server` | `snapshot.api_server.ready` | Yes |
| `node_inventory` | `snapshot.nodes[]` | Yes |
| `gpu_inventory` | `snapshot.nodes[].gpu_count` | Yes |
| `gpu_operator` | `snapshot.components.gpu_operator.ready` | No |
| `dcgm_exporter` | `snapshot.components.dcgm_exporter.ready` | Yes |
| `prometheus` | `snapshot.components.prometheus.ready` | Yes |
| `batch_scheduler` | `snapshot.components.{kueue,volcano}.ready` | No |
| `rbac_permissions` | `snapshot.permissions.*` | Yes |

**Snapshot load timing** is recorded in `snapshot["_load_latency_ms"]` for observability.

#### Kubernetes Connector (`kubernetes.py`)

The real connector imports the `kubernetes` Python client **at check time** (not at import time). This ensures that mock mode never requires `kubernetes` to be installed.

**Authentication flow:**

```
in_cluster=True  ->  config.load_incluster_config()     # ServiceAccount inside pod
in_cluster=False ->  config.load_kube_config(context=, config_file=)  # Local kubeconfig
```

If authentication fails, a single failing `CheckItem` is returned with remediation guidance, and no further checks are attempted.

**Checks executed (8 total):**

| # | Check | K8s API | Description |
|---|-------|---------|-------------|
| 1 | `api_server` | `VersionApi.get_code()` | Server reachability, git version |
| 2 | `rbac_permissions` | `AuthorizationV1Api.create_self_subject_access_review()` | Verifies `list nodes`, `list pods`, `list crds` |
| 3 | `node_inventory` | `CoreV1Api.list_node()` | Node count, Ready status, capacity |
| 4 | `gpu_inventory` | `CoreV1Api.list_node()` | `nvidia.com/gpu` capacity, mock GPU labels |
| 5 | `gpu_operator` | `CoreV1Api.list_namespaced_pod()` | Searches `gpu-operator`, `nvidia-gpu-operator` namespaces |
| 6 | `dcgm_exporter` | `CoreV1Api.list_namespaced_pod()` | Searches `gpu-operator`, `nvidia-gpu-operator`, `monitoring` |
| 7 | `batch_scheduler` | `ApiextensionsV1Api.read_custom_resource_definition()` | Checks for Kueue `clusterqueues` & Volcano `queues` CRDs |
| 8 | `prometheus` | `CoreV1Api.list_namespaced_service()` | Discovers Prometheus services in common namespaces |

**Timing wrapper:** Every check is wrapped in `_timed()` which:
- Records wall-clock latency via `time.perf_counter()`.
- Catches `ApiException` and generic `Exception` per check, returning a structured `FAIL` `CheckItem` with remediation guidance.
- Sets `latency_ms` on the result.

**Component pod discovery strategy:**
- Iterates multiple namespaces and label selectors.
- Reports all matching pods with their namespace, name, phase, and container readiness.
- If `optional=True` (GPU operator), a missing component produces `WARN`. If `optional=False` (DCGM exporter), it produces `FAIL`.

---

## 3. Request Lifecycle (End-to-End Flow)

### Example: Run checks on a cluster

```
Client
  |
  | POST /api/v1/clusters/{id}/checks
  | X-Correlation-ID: (optional)
  v
[1] CORSMiddleware
  | - Checks Origin against allowlist
  | - Handles OPTIONS preflight
  v
[2] RateLimitMiddleware
  | - Extracts client IP
  | - Checks in-memory sliding window (per-minute, per-hour)
  | - Returns 429 if exceeded, adds X-RateLimit-* headers otherwise
  v
[3] APIKeyAuthMiddleware
  | - If GPUOPT_API_KEY is set, validates X-API-Key header
  | - Skips check if path is public
  | - Returns 401/403 if invalid
  v
[4] RequestLoggingMiddleware
  | - Generates correlation ID (or uses client-provided)
  | - Logs request_started {method, path, client_host}
  v
[5] api.py: run_cluster_check(cluster_id, service)
  | - FastAPI validates cluster_id as UUID (422 if invalid)
  | - Injects EnvironmentCheckService via Depends(get_check_service)
  v
[6] services.py: check_service.check_cluster(cluster_id)
  | - repository.get_cluster(cluster_id)
  |   - Acquires RLock
  |   - SELECT * FROM clusters WHERE id=?
  |   - Returns ClusterRecord or None
  | - Raises KeyError if not found -> HTTP 404
  |
  | - build_connector(cluster) -> KubernetesConnector or MockConnector
  |
  | - CHECK_DURATION.labels(...).time():        # Context manager, captures wall time
  |     connector.run_checks()
  |     - K8s: 8 timed checks vs live API
  |     - Mock: 8 checks vs JSON snapshot
  |
  | - _overall_status(checks)                   # FAIL > WARN > SKIP > PASS
  | - Build EnvironmentCheckReport
  | - repository.save_report(report)
  |   - INSERT INTO check_reports (...)
  |
  | - CHECK_RUNS.labels(...).inc()              # Prometheus counter
  | - CLUSTER_STATUS.labels(...).set(...)        # Prometheus gauge (0-3)
  |
  | - Return EnvironmentCheckReport
  v
[7] FastAPI serializes report to JSON via Pydantic
  v
[8] RequestLoggingMiddleware logs request_completed {status_code, duration_ms}
  v
[9] Client receives 200 OK with X-Correlation-ID and X-Response-Time headers
```

---

## 4. Data Model

### Pydantic Models (in `schemas.py`)

```
ConnectorType (StrEnum)
    MOCK = "mock"
    KUBERNETES = "kubernetes"

CheckStatus (StrEnum)
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"

ClusterCreate (BaseModel)             # Input schema for POST/PUT
    + name: str (2-120 chars)
    + environment: str (default "development", max 40)
    + connector_type: ConnectorType
    + description: str | None (max 500)
    + kube_context: str | None
    + kubeconfig_path: str | None     # Validated: rejects inline YAML
    + in_cluster: bool (default False)
    + credential_ref: str | None
    + options: dict (default {})

ClusterRecord (ClusterCreate)          # Output / DB model
    + id: UUID (auto-generated)
    + created_at: datetime (UTC)
    + updated_at: datetime (UTC)

CheckItem (BaseModel)                  # Individual check result
    + name: str
    + status: CheckStatus
    + message: str
    + latency_ms: float | None
    + details: dict
    + remediation: str | None

EnvironmentCheckReport (BaseModel)     # Full check report
    + id: UUID
    + cluster_id: UUID
    + cluster_name: str
    + environment: str
    + started_at: datetime
    + completed_at: datetime
    + overall_status: CheckStatus
    + checks: list[CheckItem]
    + summary: dict[str, int]          # {pass: N, warn: N, fail: N, skip: N}

EnvironmentSummary (BaseModel)         # Aggregated health
    + generated_at: datetime
    + clusters: int
    + healthy: int
    + warning: int
    + failing: int
    + unchecked: int
    + environments: dict[str, dict]    # {"sandbox": {"clusters": 1, "healthy": 1, ...}}
```

### Relationship Diagram

```
ClusterRecord (1) ----< (N) EnvironmentCheckReport
    |                                         |
    | id: UUID                                | cluster_id: UUID
    | name: str (unique)                      | overall_status: CheckStatus
    | environment: str                        | checks: list[CheckItem]
    | connector_type: ConnectorType           | summary: dict
    | options: dict                           |
    | ...                                     | -- stored as report_json TEXT
```

---

## 5. Module Interaction Map

```
main.py
  |-- Creates FastAPI app
  |-- Registers exception handlers (exceptions.py)
  |-- Adds middleware stack (CORS -> RateLimit -> APIKeyAuth -> RequestLogging)
  |-- Includes API router
  |-- Lifespan: initializes logging, creates repository (auto-creates tables)

api.py
  |-- Depends on: dependencies.get_repository(), dependencies.get_check_service()
  |-- Uses: schemas.py models for request/response
  |-- Uses: repository.RepositoryError for 409 mapping
  |-- Uses: config.get_settings() for /health/detailed and /api/v1/info

services.py
  |-- Depends on: repository.ClusterRepository
  |-- Uses: connectors.factory.build_connector()
  |-- Uses: schemas (CheckItem, CheckStatus, EnvironmentCheckReport, EnvironmentSummary)
  |-- Emits: prometheus_client metrics (Counter, Histogram, Gauge)

repository.py
  |-- Depends on: config.get_settings() (via constructor)
  |-- Uses: schemas (ClusterCreate, ClusterRecord, EnvironmentCheckReport)
  |-- Uses: sqlite3, threading.RLock, json

connectors/factory.py
  |-- Uses: schemas.ClusterRecord, schemas.ConnectorType
  |-- Routes to: kubernetes.KubernetesConnector or mock.MockConnector

connectors/kubernetes.py
  |-- Extends: base.ClusterConnector
  |-- Uses: schemas (CheckItem, CheckStatus)
  |-- Defers: import kubernetes (at check time)

connectors/mock.py
  |-- Extends: base.ClusterConnector
  |-- Uses: schemas (CheckItem, CheckStatus)
  |-- Reads: JSON snapshot files from disk

middleware.py
  |-- Uses: config.get_settings() for API key config
  |-- Uses: uuid, time, logging

exceptions.py
  |-- Defines: GPUOptException hierarchy
  |-- Registers: FastAPI exception handlers

ratelimit.py
  |-- Uses: config.get_settings() for rate limits
  |-- In-memory: dict[str, list[float]] sliding window per IP
```

---

## 6. Middleware Pipeline Detail

### RequestLoggingMiddleware

```
Request IN
  |-- Read or generate X-Correlation-ID
  |-- Store in request.state.correlation_id
  |-- Log "request_started" (method, path, query, client, correlation_id)
  |-- time.perf_counter()
  v
  ... call_next ...
  v
  |-- latency = time.perf_counter() - start
  |-- Log "request_completed" (method, path, status_code, duration_ms, correlation_id)
  |-- Set response headers: X-Correlation-ID, X-Response-Time
Response OUT
```

### APIKeyAuthMiddleware

```
Request IN
  |-- Is GPUOPT_API_KEY set? (No -> skip, pass through)
  |-- Is path in PUBLIC_PATHS? (Yes -> skip, pass through)
  |-- Read request.headers[GPUOPT_API_KEY_HEADER]
  |   |-- Missing -> 401 {"error": "Missing API key", "header": "X-API-Key"}
  |   |-- Wrong    -> 403 {"error": "Invalid API key"}
  |   |-- Match    -> pass through
Response OUT
```

### RateLimitMiddleware

```
Request IN
  |-- Is path in SKIP_RATE_LIMIT_PATHS? (Yes -> skip, pass through)
  |-- Extract client IP
  |-- Cleanup stale entries (every 5 minutes)
  v
  |-- Check per-minute window (default 120)
  |   |-- Exceeded? -> 429 + Retry-After header
  v
  |-- Check per-hour window (default 5000)
  |   |-- Exceeded? -> 429 + Retry-After header
  v
  |-- Set X-RateLimit-Limit, X-RateLimit-Remaining on response
Response OUT
```

---

## 7. Configuration System

Settings are managed by `pydantic-settings` with the `GPUOPT_` prefix. Resolution order:

1. Default values in the `Settings` class.
2. `.env` file in the project root.
3. Environment variables (highest priority).

**Full settings table:**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GPUOPT_ENV` | str | `development` | Runtime environment label |
| `GPUOPT_DATABASE_PATH` | Path | `./data/gpuopt.db` | SQLite file path |
| `GPUOPT_LOG_LEVEL` | str | `INFO` | Logging verbosity |
| `GPUOPT_ALLOW_MOCK_GPU` | bool | `True` | Accept mock GPU labels |
| `GPUOPT_CHECK_TIMEOUT_SECONDS` | int | `15` | Per-check timeout |
| `GPUOPT_API_HOST` | str | `0.0.0.0` | Bind address |
| `GPUOPT_API_PORT` | int | `8080` | Listen port |
| `GPUOPT_CORS_ORIGINS` | list | `["*"]` | CORS allowed origins |
| `GPUOPT_API_KEY` | str | `None` | Optional API key |
| `GPUOPT_API_KEY_HEADER` | str | `X-API-Key` | Header name for API key |
| `GPUOPT_RATE_LIMIT_PER_MINUTE` | int | `120` | Max requests/minute/IP |
| `GPUOPT_RATE_LIMIT_PER_HOUR` | int | `5000` | Max requests/hour/IP |

---

## 8. Deployment Topologies

### Local Development (No K8s)

```
.venv/  +--  uvicorn main:app --reload  +--  SQLite (./data/gpuopt.db)
        +--  MockConnector              +--  Mock JSON snapshots
        +--  Prometheus /metrics
```

### Docker Compose

```
docker-compose.yml
    |
    +-- api:8080  (GPUOpt + Uvicorn + SQLite volume)
    +-- prometheus:9090  (scrapes api:8080/metrics)
```

### Kind Cluster (Local K8s)

```
kind cluster (3 nodes: 1 control-plane, 2 workers)
    |
    +-- gpuopt-system/namespace
        +-- gpuopt-backend deployment  (in-cluster config, K8s connector)
        +-- mock-dcgm-exporter deployment  (mock GPU metrics)
        +-- ServiceAccount + ClusterRole (read-only RBAC)
```

### Production K8s

```
Production cluster(s)
    |
    +-- gpuopt-system/namespace
        +-- gpuopt-backend deployment
        +-- ServiceAccount + ClusterRoleBinding
        +-- ConfigMap (env vars)
        +-- PostgreSQL (external or Cloud SQL)
        +-- Prometheus Operator ServiceMonitor
```

---

## 9. Security Model

| Layer | Control | Details |
|-------|---------|---------|
| **K8s RBAC** | Read-only ClusterRole | `list nodes/pods/crds`, `create selfsubjectaccessreviews`. No mutation. |
| **API Auth** | Optional API key | Header-based (`X-API-Key`). Disabled by default. |
| **Rate Limiting** | Per-IP sliding window | Configurable limits per minute and per hour. Health paths exempt. |
| **CORS** | Origin allowlist | Configurable via `GPUOPT_CORS_ORIGINS`. Defaults to `["*"]`. |
| **Input Validation** | Pydantic + FastAPI | All inputs validated at the boundary. Inline kubeconfig data rejected. |
| **Error Handling** | Structured JSON | No stack traces leaked. All exceptions logged server-side. |
| **Container** | Non-root user | Dockerfile runs as UID 10001 (gpuopt user). |

---

## 10. Production Roadmap

Beyond the sandbox, the following layers are planned:

```
+--------------------------+
|  Recommendation Engine   |  <-- Scoring, policy, what-if analysis
+--------------------------+
|  Digital Twin Service    |  <-- Cluster state mirroring, drift detection
+--------------------------+
|  Predictive Scheduling   |  <-- ML-based pod placement recommendations
+--------------------------+
|  Actuation Pipeline      |  <-- Approved mutation execution with rollback
+--------------------------+
|  GPUOpt Sandbox (this)   |  <-- Read-only readiness checks (current)
+--------------------------+
```

Each future layer must pass staging acceptance for:
- Policy checks
- Approval workflows
- Action idempotency
- Rollback controls
- Audit logging

---

## 11. Glossary

| Term | Definition |
|------|------------|
| **Connector** | An object that implements `ClusterConnector.run_checks()` for a specific backend (mock, K8s) |
| **CheckItem** | A single readiness check result with name, status, message, latency, details, and remediation |
| **Environment** | A logical grouping label for clusters (`sandbox`, `development`, `staging`, `production`) |
| **ClusterRecord** | A persisted cluster registration with authentication configuration |
| **Check Report** | A snapshot of all checks run against a cluster at a point in time |
| **DCGM** | NVIDIA Data Center GPU Manager -- provides GPU telemetry |
| **Kueue / Volcano** | Kubernetes batch scheduling frameworks for AI/ML workloads |
| **Mock GPU** | Synthetic GPU labels (`gpuopt.ai/mock-gpu-count`) used when no real NVIDIA hardware is available |
| **CRD** | Custom Resource Definition -- Kubernetes extension API |
