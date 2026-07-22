# Configuration Reference

GPUOpt Backend Sandbox uses environment variables for all configuration. Settings are loaded via `pydantic-settings` with the prefix `GPUOPT_` and are case-insensitive.

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GPUOPT_ENV` | string | `"development"` | Application environment (`sandbox`, `development`, `staging`, `production`) |
| `GPUOPT_DATABASE_PATH` | path | `./data/gpuopt.db` | SQLite database file path. Parent directory is auto-created. |
| `GPUOPT_LOG_LEVEL` | string | `"INFO"` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `GPUOPT_ALLOW_MOCK_GPU` | boolean | `true` | Allow mock GPU labels when no real `nvidia.com/gpu` resources exist |
| `GPUOPT_CHECK_TIMEOUT_SECONDS` | integer | `15` | Timeout for individual environment checks |
| `GPUOPT_API_HOST` | string | `"0.0.0.0"` | Uvicorn bind address |
| `GPUOPT_API_PORT` | integer | `8080` | Uvicorn listen port |
| `GPUOPT_CORS_ORIGINS` | list | `["*"]` | Allowed CORS origins |
| `GPUOPT_API_KEY` | string | `null` | Optional API key for authentication (empty = disabled) |
| `GPUOPT_API_KEY_HEADER` | string | `"X-API-Key"` | HTTP header name for API key |
| `GPUOPT_RATE_LIMIT_PER_MINUTE` | integer | `120` | Max requests per minute per IP |
| `GPUOPT_RATE_LIMIT_PER_HOUR` | integer | `5000` | Max requests per hour per IP |

## Configuration Sources

Settings are resolved in this order (last wins):

1. **Default values** in `Settings` class.
2. **`.env` file** in the project root (loaded automatically).
3. **Environment variables** with `GPUOPT_` prefix.

## Example `.env` File

```bash
GPUOPT_ENV=development
GPUOPT_DATABASE_PATH=./data/gpuopt.db
GPUOPT_LOG_LEVEL=INFO
GPUOPT_ALLOW_MOCK_GPU=true
GPUOPT_CHECK_TIMEOUT_SECONDS=15
GPUOPT_API_HOST=0.0.0.0
GPUOPT_API_PORT=8080
GPUOPT_CORS_ORIGINS=["*"]
GPUOPT_API_KEY=
GPUOPT_API_KEY_HEADER=X-API-Key
GPUOPT_RATE_LIMIT_PER_MINUTE=120
GPUOPT_RATE_LIMIT_PER_HOUR=5000
```

## Environment-Specific Configuration

### Local Development (no Kubernetes)

```bash
GPUOPT_ENV=sandbox
GPUOPT_ALLOW_MOCK_GPU=true
GPUOPT_LOG_LEVEL=DEBUG
```

### Docker Compose

Environment variables are set directly in `docker-compose.yml`:

```yaml
services:
  api:
    environment:
      GPUOPT_ENV: sandbox
      GPUOPT_DATABASE_PATH: /data/gpuopt.db
      GPUOPT_ALLOW_MOCK_GPU: "true"
```

### Kubernetes Deployment

Configuration is injected via a ConfigMap in `infra/k8s/base/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gpuopt-config
  namespace: gpuopt-system
data:
  GPUOPT_ENV: development
  GPUOPT_DATABASE_PATH: /data/gpuopt.db
  GPUOPT_ALLOW_MOCK_GPU: "true"
  GPUOPT_LOG_LEVEL: INFO
```

The Deployment references this ConfigMap via `envFrom`.

## Cluster Registration Options

Clusters are registered via YAML files or API calls. The `options` field is connector-specific:

### Mock Connector Options

| Key | Type | Description |
|-----|------|-------------|
| `snapshot_path` | string | Path to a JSON snapshot file for mock cluster state |

### Kubernetes Connector Options

| Key | Type | Description |
|-----|------|-------------|
| `allow_mock_gpu` | boolean | Accept mock GPU labels instead of real `nvidia.com/gpu` resources |

### Example: Mock Cluster

```yaml
clusters:
  - name: local-mock
    environment: sandbox
    connector_type: mock
    description: Local no-GPU validation using a synthetic snapshot.
    options:
      snapshot_path: sandbox/mock-clusters/local-kind.json
```

### Example: Kubernetes Cluster

```yaml
clusters:
  - name: local-kind
    environment: development
    connector_type: kubernetes
    description: Local kind cluster with GPU labels and mock telemetry.
    kube_context: kind-gpuopt
    kubeconfig_path: ~/.kube/config
    options:
      allow_mock_gpu: true
```

### Example: Real GPU Cluster

```yaml
clusters:
  - name: staging-gpu
    environment: staging
    connector_type: kubernetes
    description: Real NVIDIA GPU staging cluster.
    kube_context: gpuopt-staging
    kubeconfig_path: ~/.kube/config
    credential_ref: local-kubeconfig-context
    options:
      allow_mock_gpu: false
```

## Mock Snapshot Format

The JSON snapshot used by the Mock Connector:

```json
{
  "api_server": { "ready": true, "version": "kind-sandbox" },
  "nodes": [
    { "name": "gpuopt-control-plane", "ready": true, "gpu_count": 0 },
    { "name": "gpuopt-worker", "ready": true, "gpu_count": 4, "gpu_model": "mock-a100" },
    { "name": "gpuopt-worker2", "ready": true, "gpu_count": 4, "gpu_model": "mock-l40s" }
  ],
  "components": {
    "gpu_operator": { "ready": false, "mode": "mock-labels-only" },
    "dcgm_exporter": { "ready": true, "metrics_endpoint": "http://mock-dcgm-exporter:9400/metrics" },
    "prometheus": { "ready": true },
    "kueue": { "ready": true },
    "volcano": { "ready": false }
  },
  "permissions": {
    "list_nodes": true,
    "list_pods": true,
    "read_crds": true
  }
}
```

## CLI Configuration

The CLI reads the same environment variables. It does not require a running server.

```bash
# Seed clusters from a YAML file
python -m gpuopt.cli seed --file environments.mock.yaml

# Run checks and output JSON
python -m gpuopt.cli check-all --file environments.mock.yaml --json
```
