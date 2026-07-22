# Contributing

## Getting Started

1. Fork and clone the repository.
2. Set up the development environment:

```bash
cd gpuopt-backend-sandbox
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

3. Run the tests to confirm everything works:

```bash
make test
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feat/my-feature
```

### 2. Make Changes

- Follow the existing code patterns.
- Add tests for new functionality.
- Keep changes focused and minimal.

### 3. Run Checks

```bash
make lint         # Ruff linter
make test         # Pytest
```

### 4. Commit

Follow conventional commit messages:

```
feat: add new check for network policies
fix: handle missing kubeconfig gracefully
docs: update API reference for new endpoint
test: add integration test for cluster upsert
chore: bump dependencies
refactor: extract common check logic
```

### 5. Open a Pull Request

- Describe what the PR does and why.
- Reference any related issues.
- Ensure all checks pass.

## Code Style

- Line length: 100 characters (enforced by Ruff).
- Type hints on all public functions.
- Pydantic models for all API request/response schemas.
- Docstrings only when the purpose is non-obvious.
- No comments that restate what the code does.

## Testing

- Tests live in `tests/`.
- Use the `client` fixture for API integration tests.
- Each test gets a fresh temporary SQLite database.
- Test the happy path AND error cases.

### Adding a Test

```python
def test_my_new_feature(client):
    response = client.get("/api/v1/my-endpoint")
    assert response.status_code == 200
    assert response.json()["status"] == "expected"
```

## Project Structure Rules

- `src/gpuopt/` — application source code only.
- `tests/` — test files only.
- `infra/` — Kubernetes manifests and infrastructure config.
- `scripts/` — shell scripts for automation.
- `docs/` — documentation in markdown.
- `sandbox/` — mock data and test fixtures.

## Adding Dependencies

1. Add to `dependencies` in `pyproject.toml` (runtime) or `[project.optional-dependencies] dev` (dev only).
2. Run `pip install -e '.[dev]'` to install.
3. Update `Dockerfile` if the dependency needs system packages.

## Key Design Decisions

- **Read-only Kubernetes**: All connectors are non-mutating. Do not add write operations.
- **SQLite for sandbox**: Production should use PostgreSQL with migrations.
- **Connector pattern**: New cluster types (e.g., EKS, GKE) should implement `ClusterConnector` and register in the factory.
- **Pydantic validation**: All API inputs are validated at the boundary. No manual validation in route handlers.

## Reporting Issues

Use the GitHub issue tracker. Include:
- Steps to reproduce.
- Expected behavior.
- Actual behavior.
- Python version, OS, and relevant package versions.
