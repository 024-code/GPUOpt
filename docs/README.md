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

## Table of Contents

- [Quick Start Without Kubernetes](#quick-start-without-kubernetes)
- [Local Kubernetes Sandbox](#local-kubernetes-sandbox)
- [Real NVIDIA GPU Cluster](#real-nvidia-gpu-cluster)
- [Main Endpoints](#main-endpoints)
- [Security Boundary](#security-boundary)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

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

Open `http://127.0.0.1:8080/docs` to see the interactive Swagger UI.

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
| `GET` | `/health/ready` | Readiness probe (checks DB) |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/api/v1/clusters` | Register a new cluster |
| `PUT` | `/api/v1/clusters/by-name/{name}` | Upsert a cluster by name |
| `GET` | `/api/v1/clusters` | List all registered clusters |
| `GET` | `/api/v1/clusters/{id}` | Get a single cluster |
| `DELETE` | `/api/v1/clusters/{id}` | Delete a cluster |
| `POST` | `/api/v1/clusters/{id}/checks` | Run readiness checks on a cluster |
| `GET` | `/api/v1/clusters/{id}/checks/latest` | Get the latest check report |
| `POST` | `/api/v1/environments/check-all` | Run checks on all registered clusters |
| `GET` | `/api/v1/environments/summary` | Get an aggregated environment health summary |

See [docs/API.md](./API.md) for the full API reference.

## Security Boundary

The sandbox ClusterRole is read-only. Do not add mutation permissions until the recommendation engine, policy checks, approvals, action idempotency, and rollback controls have passed staging acceptance tests.

Use `environments.example.yaml` as a template for registering real development and staging contexts.

## Project Structure

```
gpuopt-backend-sandbox/
├── src/gpuopt/                  # Python package (application source)
│   ├── __init__.py              # Package version (0.1.0)
│   ├── main.py                  # FastAPI app factory with lifespan
│   ├── config.py                # Settings via pydantic-settings (env-based)
│   ├── schemas.py               # Pydantic models for API + DB
│   ├── api.py                   # All route handlers (FastAPI router)
│   ├── services.py              # Business logic: check orchestration + metrics
│   ├── repository.py            # SQLite persistence (thread-safe)
│   ├── dependencies.py          # Dependency injection (lru_cache singletons)
│   ├── cli.py                   # CLI entry point (seed, check-all)
│   └── connectors/
│       ├── base.py              # Abstract ClusterConnector interface
│       ├── factory.py           # Connector factory (mock vs kubernetes)
│       ├── mock.py              # Mock connector for laptop/CI
│       └── kubernetes.py        # Real Kubernetes connector (8 readiness checks)
├── tests/
│   ├── conftest.py              # Pytest fixtures (isolated DB per test)
│   └── test_api.py              # API integration tests
├── infra/
│   ├── k8s/base/                # Production K8s manifests
│   │   ├── namespace.yaml       # gpuopt-system namespace
│   │   ├── serviceaccount.yaml  # gpuopt-backend ServiceAccount
│   │   ├── rbac.yaml            # Read-only ClusterRole + binding
│   │   ├── configmap.yaml       # Environment variables
│   │   ├── deployment.yaml      # GPUOpt backend Deployment
│   │   └── service.yaml         # ClusterIP Service
│   ├── k8s/mock-dcgm/           # Mock DCGM telemetry for kind
│   │   ├── configmap.yaml       # Static Prometheus-format GPU metrics
│   │   ├── deployment.yaml      # Python HTTP server serving mock metrics
│   │   └── service.yaml         # Service exposing :9400
│   ├── kind/
│   │   └── kind-config.yaml     # 3-node kind cluster (1 control-plane + 2 workers)
│   └── prometheus/
│       └── prometheus.yml       # Scrape config for GPUOpt + mock DCGM
├── sandbox/mock-clusters/
│   └── local-kind.json          # Mock snapshot for kind cluster
├── scripts/
│   ├── bootstrap_kind.sh        # One-shot kind cluster setup
│   └── check_environment.sh     # Preflight check script
├── environments.mock.yaml       # Mock cluster registration template
├── environments.example.yaml    # Multi-environment registration template
├── pyproject.toml               # Build config, dependencies, scripts
├── Dockerfile                   # Production container image
├── docker-compose.yml           # Local dev: API + Prometheus
├── Makefile                     # Dev workflow commands
├── .env.example                 # Environment variable template
└── .gitignore
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Web Framework | FastAPI 0.115+ |
| Server | Uvicorn (ASGI) |
| Validation | Pydantic 2.8+, Pydantic Settings 2.4+ |
| Database | SQLite (sandbox), PostgreSQL (production roadmap) |
| K8s Client | `kubernetes` Python client 31-34 |
| Metrics | `prometheus-client` 0.20+ |
| HTTP | `httpx` 0.27+ |
| Config | `PyYAML` 6.0+ |
| Linting | Ruff 0.6+ |
| Testing | pytest 8.0+, pytest-asyncio 0.23+ |
| Container | Docker (python:3.12-slim) |
| Orchestration | Docker Compose, kind, Kubernetes |
