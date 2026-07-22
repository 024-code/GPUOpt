# Developer Guide

## Setup

```bash
cd gpuopt-backend-sandbox
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
cp .env.example .env
```

## Running

```bash
make dev          # Start uvicorn with --reload
make test         # Run pytest
make lint         # Run ruff linter
make seed         # Register mock clusters from environments.mock.yaml
make check-all    # Run readiness checks on all clusters
```

## Project Layout

```
src/gpuopt/
├── __init__.py          # Package version
├── main.py              # FastAPI app, lifespan, router inclusion
├── config.py            # Settings (pydantic-settings)
├── schemas.py           # Pydantic models + enums
├── api.py               # Route handlers
├── services.py          # Business logic + Prometheus metrics
├── repository.py        # SQLite CRUD
├── dependencies.py      # DI singletons
├── cli.py               # CLI (seed, check-all)
└── connectors/
    ├── base.py          # Abstract ClusterConnector
    ├── factory.py       # Mock vs Kubernetes routing
    ├── mock.py          # Mock connector
    └── kubernetes.py    # Real K8s connector
```

## Adding a New Check

1. Open `src/gpuopt/connectors/kubernetes.py`.
2. Add a new method:

```python
@staticmethod
def _check_my_thing(core: Any) -> CheckItem:
    # Query Kubernetes API
    # Return CheckItem with name, status, message, details
    ...
```

3. Register it in `run_checks()`:

```python
checks.append(self._timed("my_thing", lambda: self._check_my_thing(core), ApiException))
```

4. Do the same in `src/gpuopt/connectors/mock.py` if the mock connector should support it.

5. Add a test in `tests/test_api.py`.

## Adding a New Connector

1. Create `src/gpuopt/connectors/myconnector.py`:

```python
from gpuopt.schemas import CheckItem, ClusterRecord
from .base import ClusterConnector

class MyConnector(ClusterConnector):
    def run_checks(self) -> list[CheckItem]:
        return [CheckItem(name="my_check", status="pass", message="OK")]
```

2. Register in `src/gpuopt/connectors/factory.py`:

```python
from .myconnector import MyConnector

def build_connector(cluster: ClusterRecord) -> ClusterConnector:
    ...
    if cluster.connector_type == ConnectorType.MY_TYPE:
        return MyConnector(cluster)
```

3. Add the enum value to `ConnectorType` in `src/gpuopt/schemas.py`.

## Adding a New API Endpoint

1. Open `src/gpuopt/api.py`.
2. Add a route:

```python
@router.get("/api/v1/my-endpoint", tags=["custom"])
def my_endpoint(
    repository: ClusterRepository = Depends(get_repository),
) -> dict:
    return {"data": repository.list_clusters()}
```

3. Add the corresponding Pydantic model in `schemas.py` if needed.

## Testing

Tests use `pytest` with an isolated SQLite database per test (via `tmp_path`).

```bash
make test
# or: PYTHONPATH=src pytest -q
```

### Test Fixtures (`tests/conftest.py`)

- `client` — FastAPI `TestClient` with a fresh temporary database.
- All `lru_cache` singletons are cleared before and after each test.

### Running Specific Tests

```bash
pytest tests/test_api.py::test_liveness_and_readiness -v
```

### Adding Tests

```python
def test_my_new_feature(client):
    response = client.get("/api/v1/my-endpoint")
    assert response.status_code == 200
    assert "key" in response.json()
```

## Linting

```bash
make lint
# or: ruff check src tests
```

Ruff is configured with `line-length = 100`.

## CLI Usage

The CLI is registered as `gpuopt` via `pyproject.toml`:

```bash
# Seed clusters from a YAML file
gpuopt seed --file environments.mock.yaml

# Run checks on all clusters
gpuopt check-all --file environments.mock.yaml

# Output as JSON
gpuopt check-all --file environments.mock.yaml --json
```

### Exit Codes

- `0` — All checks passed.
- `1` — At least one check failed.

## Adding a New Environment

Edit `environments.mock.yaml` or `environments.example.yaml`:

```yaml
clusters:
  - name: my-cluster
    environment: development
    connector_type: kubernetes
    description: My development cluster.
    kube_context: my-context
    kubeconfig_path: ~/.kube/config
    options:
      allow_mock_gpu: true
```

Then seed it:

```bash
make seed
# or: python -m gpuopt.cli seed --file environments.mock.yaml
```

## Commit Conventions

Follow conventional commits:
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation
- `chore:` — maintenance
- `test:` — adding/updating tests
- `refactor:` — code restructuring
