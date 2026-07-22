# Deployment Guide

## Prerequisites

| Method | Requirements |
|--------|-------------|
| Local development | Python 3.11+, pip |
| Docker Compose | Docker, Docker Compose |
| kind sandbox | Docker, kubectl, kind |
| Production Kubernetes | Docker, kubectl, Kubernetes cluster, NVIDIA GPU Operator (for real GPUs) |

---

## Option 1: Local Development (No Kubernetes)

```bash
cd gpuopt-backend-sandbox
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
cp .env.example .env
make test                        # Run tests
make seed                        # Register mock clusters
make check-all                   # Run checks
make dev                         # Start dev server
```

Server starts at `http://127.0.0.1:8080`. Swagger UI at `/docs`.

---

## Option 2: Docker Compose

```bash
# Start API + Prometheus
make compose-up

# Or manually:
docker compose up --build -d
```

**Services:**
- `api` — GPUOpt backend on port `8080`
- `prometheus` — Prometheus on port `9090`, scraping GPUOpt + mock DCGM

**Stop and clean up:**

```bash
make compose-down
# or: docker compose down -v
```

**Persistent data:**

The SQLite database is stored in a Docker volume (`gpuopt-data`) mounted at `/data/gpuopt.db`.

---

## Option 3: kind Cluster Sandbox

### One-Command Bootstrap

```bash
make kind-up
```

This runs `scripts/bootstrap_kind.sh`, which:

1. Creates a 3-node kind cluster named `gpuopt`.
2. Labels two worker nodes with mock GPU labels (`gpuopt.ai/mock-gpu-count`, `gpuopt.ai/gpu-model`).
3. Builds and loads the GPUOpt Docker image into kind.
4. Applies namespace, RBAC, ConfigMap, and mock DCGM manifests.
5. Deploys GPUOpt backend with in-cluster credentials.
6. Waits for rollouts to complete.
7. Registers the cluster via port-forward and runs readiness checks.

### Manual kind Setup

```bash
# Create cluster
kind create cluster --name gpuopt --config infra/kind/kind-config.yaml

# Label GPU workers
kubectl --context kind-gpuopt label --overwrite \
  $(kubectl --context kind-gpuopt get nodes -o name | grep worker | head -1) \
  gpuopt.ai/mock-gpu-count=4 gpuopt.ai/gpu-model=mock-a100

# Build and load image
docker build -t gpuopt-backend-sandbox:local .
kind load docker-image --name gpuopt gpuopt-backend-sandbox:local

# Apply manifests
kubectl --context kind-gpuopt apply -f infra/k8s/base/
kubectl --context kind-gpuopt apply -f infra/k8s/mock-dcgm/

# Port-forward
kubectl --context kind-gpuopt -n gpuopt-system port-forward service/gpuopt-backend 8080:8080
```

### Accessing the API

```bash
kubectl --context kind-gpuopt -n gpuopt-system port-forward service/gpuopt-backend 8080:8080
# Open http://127.0.0.1:8080/docs
```

### Teardown

```bash
make kind-down
# or: kind delete cluster --name gpuopt
```

---

## Option 4: Production Kubernetes

### Step 1: Build and Push the Image

```bash
docker build -t your-registry/gpuopt-backend:0.1.0 .
docker push your-registry/gpuopt-backend:0.1.0
```

Update `infra/k8s/base/deployment.yaml` to reference your image.

### Step 2: Create Namespace and ServiceAccount

```bash
kubectl apply -f infra/k8s/base/namespace.yaml
kubectl apply -f infra/k8s/base/serviceaccount.yaml
```

### Step 3: Apply RBAC

```bash
kubectl apply -f infra/k8s/base/rbac.yaml
```

This creates a **read-only** ClusterRole with permissions for:
- `get`, `list`, `watch` on nodes, pods, services, endpoints, events, namespaces
- `get`, `list`, `watch` on deployments, daemonsets, statefulsets
- `get`, `list` on customresourcedefinitions
- `create` on selfsubjectaccessreviews

### Step 4: Configure

Edit `infra/k8s/base/configmap.yaml` for your environment:

```yaml
data:
  GPUOPT_ENV: production
  GPUOPT_DATABASE_PATH: /data/gpuopt.db
  GPUOPT_ALLOW_MOCK_GPU: "false"
  GPUOPT_LOG_LEVEL: WARNING
```

```bash
kubectl apply -f infra/k8s/base/configmap.yaml
```

### Step 5: Deploy

```bash
kubectl apply -f infra/k8s/base/deployment.yaml
kubectl apply -f infra/k8s/base/service.yaml
```

### Step 6: Verify

```bash
kubectl -n gpuopt-system rollout status deployment/gpuopt-backend
kubectl -n gpuopt-system port-forward service/gpuopt-backend 8080:8080
curl http://127.0.0.1:8080/health/ready
```

### Step 7: Register Real Clusters

```bash
curl -X POST http://127.0.0.1:8080/api/v1/clusters \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "prod-gpu",
    "environment": "production",
    "connector_type": "kubernetes",
    "description": "Production GPU cluster",
    "in_cluster": true,
    "options": {"allow_mock_gpu": false}
  }'
```

---

## Kubernetes Manifests Reference

| File | Purpose |
|------|---------|
| `infra/k8s/base/namespace.yaml` | `gpuopt-system` namespace with `app.kubernetes.io/part-of: gpuopt` label |
| `infra/k8s/base/serviceaccount.yaml` | `gpuopt-backend` ServiceAccount |
| `infra/k8s/base/rbac.yaml` | Read-only ClusterRole + ClusterRoleBinding |
| `infra/k8s/base/configmap.yaml` | Environment variables for the backend |
| `infra/k8s/base/deployment.yaml` | 1-replica Deployment with security hardening |
| `infra/k8s/base/service.yaml` | ClusterIP Service on port 8080 |
| `infra/k8s/mock-dcgm/configmap.yaml` | Static DCGM metrics in Prometheus format |
| `infra/k8s/mock-dcgm/deployment.yaml` | Mock DCGM exporter (Python HTTP server on :9400) |
| `infra/k8s/mock-dcgm/service.yaml` | Service exposing mock DCGM on port 9400 |

## Security Hardening (Deployment)

The production Deployment includes:

- `runAsNonRoot: true` with UID `10001`
- `readOnlyRootFilesystem: true`
- `allowPrivilegeEscalation: false`
- `capabilities: drop: ["ALL"]`
- Resource limits: 1 CPU, 512Mi memory
- Liveness probe: `GET /health/live` every 10s
- Readiness probe: `GET /health/ready` every 5s

---

## Health Checks

```bash
# Liveness
curl http://127.0.0.1:8080/health/live
# {"status": "alive"}

# Readiness
curl http://127.0.0.1:8080/health/ready
# {"status": "ready", "registered_clusters": 2}
```
