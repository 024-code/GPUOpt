# GPUOpt Backend Sandbox

A runnable backend starter for registering Kubernetes environments and checking whether each cluster is ready for GPU optimization work.

## What This Sandbox Proves

- **Cluster registration and environment separation**: sandbox, development, staging, and production.
- **Read-only Kubernetes connectivity** using kubeconfig or an in-cluster ServiceAccount.
- **Platform readiness checks**: API server, RBAC, node, GPU, NVIDIA GPU Operator, DCGM exporter, Prometheus, and Kueue/Volcano.
- **Mock GPU mode** for laptop/CI environments without NVIDIA hardware.
- **SQLite persistence** for cluster records and check history.
- **Prometheus metrics**, health probes, OpenAPI documentation, and Kubernetes manifests.

This is the engineering foundation only. Predictive models, the digital twin, recommendation scoring, and production actuation are later roadmap phases.

## Quick Start Without Kubernetes

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
make test
make seed
make check-all
make dev
```

Open `http://127.0.0.1:8080/docs`.

## Local Kubernetes Sandbox

Prerequisites: Docker, kubectl, and kind.

```bash
make kind-up
```

The script creates a three-node kind cluster, labels two workers as mock GPU nodes, deploys mock DCGM metrics, deploys GPUOpt with read-only RBAC, registers the cluster, and executes the readiness checks.

## Real NVIDIA GPU Cluster

1. Install the NVIDIA GPU Operator or the required driver/device-plugin stack.
2. Confirm nodes advertise `nvidia.com/gpu` in capacity and allocatable resources.
3. Confirm DCGM exporter exposes `/metrics` and is scraped by Prometheus.
4. Apply `infra/k8s/base/namespace.yaml`, `serviceaccount.yaml`, and `rbac.yaml`.
5. Deploy GPUOpt using `in_cluster=true`, or run it externally with a read-only kubeconfig context.
6. Set `options.allow_mock_gpu=false` when registering the real cluster.

## Main Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe |
| `GET` | `/health/detailed` | Detailed health with system info and cluster counts |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/api/v1/info` | System information and links |
| `POST` | `/api/v1/clusters` | Register a cluster |
| `PUT` | `/api/v1/clusters/by-name/{name}` | Upsert a cluster by name |
| `GET` | `/api/v1/clusters` | List all clusters |
| `GET` | `/api/v1/clusters/{id}` | Get a cluster |
| `DELETE` | `/api/v1/clusters/{id}` | Delete a cluster |
| `POST` | `/api/v1/clusters/{id}/checks` | Run checks on a cluster |
| `GET` | `/api/v1/clusters/{id}/checks/latest` | Get latest check report |
| `POST` | `/api/v1/environments/check-all` | Run checks on all clusters |
| `GET` | `/api/v1/environments/summary` | Environment health summary |

## Middleware & Security

| Feature | Description | Configuration |
|---------|-------------|---------------|
| Request Logging | Correlation IDs (`X-Correlation-ID`), response times (`X-Response-Time`) | Always enabled |
| Rate Limiting | Per-IP limits (configurable per minute/hour) | `GPUOPT_RATE_LIMIT_PER_MINUTE`, `GPUOPT_RATE_LIMIT_PER_HOUR` |
| API Key Auth | Optional header-based (`X-API-Key` by default) | `GPUOPT_API_KEY`, `GPUOPT_API_KEY_HEADER` |
| CORS | Configurable origin allowlist | `GPUOPT_CORS_ORIGINS` |
| Error Handling | Structured JSON error responses | Always enabled |

## Migrations

```bash
make migrate
# or: PYTHONPATH=src python scripts/migrate.py
```

Applies pending SQL migration files from `migrations/`. For production, replace with Alembic + PostgreSQL.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Full project overview and structure |
| [docs/API.md](docs/API.md) | Complete API reference with examples |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and data flow |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | All environment variables and options |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Local, Docker Compose, kind, and production K8s deployment |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Developer guide, adding checks, connectors, endpoints |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Operational runbook and troubleshooting |
| [docs/SECURITY.md](docs/SECURITY.md) | Security model, RBAC, container hardening |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | How to contribute |

## Security Boundary

The sandbox ClusterRole is read-only. Do not add mutation permissions until the recommendation engine, policy checks, approvals, action idempotency, and rollback controls have passed staging acceptance tests.

Use `environments.example.yaml` as a template for registering real development and staging contexts.

## Project Structure

```
gpuopt-backend-sandbox/
├── src/gpuopt/                  # Application source
│   ├── main.py                  # FastAPI app factory
│   ├── config.py                # Environment-based settings
│   ├── schemas.py               # Pydantic models + enums
│   ├── api.py                   # Route handlers
│   ├── services.py              # Check orchestration + metrics
│   ├── repository.py            # SQLite persistence
│   ├── dependencies.py          # Dependency injection
│   ├── cli.py                   # CLI entry point
│   └── connectors/              # Cluster connectors
│       ├── base.py              # Abstract interface
│       ├── factory.py           # Connector routing
│       ├── mock.py              # Mock connector
│       └── kubernetes.py        # Real K8s connector
├── tests/                       # Integration tests
├── infra/                       # K8s manifests + Prometheus config
├── scripts/                     # Shell automation scripts
├── sandbox/                     # Mock cluster snapshots
├── docs/                        # Documentation
├── pyproject.toml               # Build config + dependencies
├── Dockerfile                   # Production container
├── docker-compose.yml           # Local dev stack
└── Makefile                     # Dev workflow commands
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Framework | FastAPI 0.115+ |
| Server | Uvicorn (ASGI) |
| Validation | Pydantic 2.8+ |
| Database | SQLite (sandbox), PostgreSQL (production) |
| K8s Client | `kubernetes` Python client 31-34 |
| Metrics | `prometheus-client` 0.20+ |
| Container | Docker (python:3.12-slim) |
| Orchestration | Docker Compose, kind, Kubernetes |
